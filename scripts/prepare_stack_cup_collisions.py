"""Regenerate the three task-local hollow cup colliders used by stack_papercup.

Run with an Isaac Sim Python environment providing pxr (including schema plugins),
NumPy, SciPy and trimesh. Original models and task YAML files are never modified.
The derived assets reference the original visual meshes, disable their convex-hull
colliders, and use a closed, 1 mm voxel-repaired surface with PhysX SDF collision.
"""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy import ndimage
import trimesh
from pxr import Usd,UsdGeom,UsdPhysics,Gf,Sdf

ROOT=Path(__file__).resolve().parents[1]
dest=ROOT/'models/task_variants/stack_cups';dest.mkdir(parents=True,exist_ok=True)
records={};pitch=.001
for name in ['white_100mmcup','white_80mmcup','white_75mmcup']:
 source=ROOT/'models/all_object'/f'{name}.usd';stage=Usd.Stage.Open(str(source))
 meshes=[]
 for prim in stage.Traverse():
  if not prim.IsA(UsdGeom.Mesh):continue
  mesh=UsdGeom.Mesh(prim);points=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
  transform=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
  points=(np.c_[points,np.ones(len(points))]@transform)[:,:3]
  ids=np.asarray(mesh.GetFaceVertexIndicesAttr().Get());tri=[];idx=0
  for count in mesh.GetFaceVertexCountsAttr().Get():
   poly=ids[idx:idx+count];idx+=count
   tri.extend([poly[0],poly[j],poly[j+1]] for j in range(1,count-1))
  meshes.append(trimesh.Trimesh(points,np.asarray(tri),process=True))
 visual=trimesh.util.concatenate(meshes)
 grid=visual.voxelized(pitch)
 matrix=np.pad(grid.matrix,3);origin=grid.transform[:3,3]-3*pitch
 # Close only voxel-scale surface gaps. The open mouth remains connected to
 # the exterior, so flood filling fills the wall, not the cup's cavity.
 matrix=ndimage.binary_closing(matrix,iterations=1)
 matrix=ndimage.binary_fill_holes(matrix)
 labels,n=ndimage.label(matrix);sizes=np.bincount(labels.ravel());sizes[0]=0
 matrix=labels==int(sizes.argmax())
 collision=trimesh.voxel.ops.matrix_to_marching_cubes(matrix,pitch=pitch)
 collision.apply_translation(origin)
 if not collision.is_watertight:raise RuntimeError(f'{name}: collision mesh is not closed')
 center=(visual.bounds[0,:2]+visual.bounds[1,:2])/2
 testpoints=np.array([[*center,z] for z in np.linspace(visual.bounds[0,2]+.025,visual.bounds[1,2]-.005,15)])
 indices=np.rint((testpoints-origin)/pitch).astype(int)
 cavity_occupancy=[bool(matrix[tuple(i)]) for i in indices]
 if any(cavity_occupancy[-5:]):raise RuntimeError(f'{name}: cup mouth is sealed: {cavity_occupancy}')
 file=dest/f'{name}_hollow.usda'
 if file.exists():file.unlink()  # Regenerate only this task-owned collision copy.
 derived=Usd.Stage.CreateNew(str(file))
 root=UsdGeom.Xform.Define(derived,'/World').GetPrim();derived.SetDefaultPrim(root)
 root.GetReferences().AddReference('../../all_object/'+name+'.usd')
 # The spawner can override mass only when the mass schema exists.
 UsdPhysics.MassAPI.Apply(root).CreateMassAttr(.1)
 for prim in stage.Traverse():
  if prim.HasAPI(UsdPhysics.CollisionAPI):
   override=derived.OverridePrim(prim.GetPath());UsdPhysics.CollisionAPI(override).CreateCollisionEnabledAttr(False)
 collider=UsdGeom.Mesh.Define(derived,'/World/StackCupCollision')
 collider.CreatePointsAttr([Gf.Vec3f(*map(float,p)) for p in collision.vertices])
 collider.CreateFaceVertexCountsAttr([3]*len(collision.faces))
 collider.CreateFaceVertexIndicesAttr(collision.faces.ravel().tolist())
 collider.CreateSubdivisionSchemeAttr('none');collider.CreateVisibilityAttr('invisible')
 UsdPhysics.CollisionAPI.Apply(collider.GetPrim()).CreateCollisionEnabledAttr(True)
 UsdPhysics.MeshCollisionAPI.Apply(collider.GetPrim()).CreateApproximationAttr('sdf')
 collider.GetPrim().AddAppliedSchema('PhysxSDFMeshCollisionAPI')
 collider.GetPrim().CreateAttribute('physxSDFMeshCollision:sdfResolution',Sdf.ValueTypeNames.UInt).Set(256)
 UsdGeom.SetStageUpAxis(derived,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(derived,1.0)
 derived.GetRootLayer().Save()
 records[name]={'original':str(source.relative_to(ROOT)),'original_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
  'variant':str(file.relative_to(ROOT)),'voxel_pitch_m':pitch,'sdf_resolution':256,'closed':collision.is_watertight,
  'collision_vertices':len(collision.vertices),'collision_faces':len(collision.faces),'cavity_centerline_occupied':cavity_occupancy,
  'bounds':visual.bounds.tolist()}
 print(name,records[name],flush=True)
(dest/'provenance.json').write_text(json.dumps(records,indent=2))

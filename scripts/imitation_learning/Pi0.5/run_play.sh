# python scripts/imitation_learning/Pi0.5/play.py --task Pick_Hang_WhiteCup_On_Rack-v0 \
#     --server_host 10.184.17.177 --server_port 8000 \
#     --prompt "Grab the white cup, hang it on the cup rack." \
#     --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/hang/hang_cup.yaml \
#     --horizon 250 --num_rollouts 30 --enable_cameras --headless

# python scripts/imitation_learning/Pi0.5/play.py --task Pick_cupwithhandle_Pour-v0 \
#     --server_host 10.184.17.177 --server_port 8008 \
#     --prompt "pick up the red mug , place it over the yellow bowl and tilt it. " \
#     --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pour/pour_mugcup.yaml \
#     --horizon 250 --num_rollouts 30 --enable_cameras --headless

# python scripts/imitation_learning/Pi0.5/play.py --task Pick_basketwithhandle_Pour-v0 \
#     --server_host 10.184.17.177 --server_port 8008 \
#     --prompt "Grab the pink ladle, place it over the yellow bowl, and tilt it." \
#     --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pour/pour_basket.yaml \
#     --horizon 250 --num_rollouts 30 --enable_cameras --headless

# python scripts/imitation_learning/Pi0.5/play.py --task Pick_redcup_in_container-v0 \
#     --server_host 10.184.17.177 --server_port 8008 \
#     --prompt "pick up the redcup , place it on the double shelf. " \
#     --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pick/pick_mugcup_place_woodshelf.yaml \
#     --horizon 250 --num_rollouts 30 --enable_cameras --headless

# python scripts/imitation_learning/Pi0.5/play.py --task Pick_Stack_Cube-v0 \
#     --server_host 10.184.17.177 --server_port 8008 \
#     --prompt "Grab the red block and place it on the blue block, then grab the green block and place it on the red block." \
#     --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/stack/stack_cube.yaml \
#     --horizon 250 --num_rollouts 30 --enable_cameras --headless

# python scripts/imitation_learning/Pi0.5/play.py --task Pick_MugCup_Place_Coffeemachine-v0 \
#     --server_host 10.184.17.177 --server_port 8008 \
#     --prompt "Grab the cup, place it in the coffeemachine." \
#     --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pick/pick_mugcup_place_coffeemachine.yaml \
#     --horizon 250 --num_rollouts 30 --enable_cameras --headless

# python scripts/imitation_learning/Pi0.5/play.py --task Pick_Hang_RedMug_On_Rack-v0 \
#     --server_host 10.184.17.177 --server_port 8008 \
#     --prompt "Grab the redmug, hang it on the cup rack." \
#     --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/hang/hang_mugcup.yaml \
#     --horizon 250 --num_rollouts 30 --enable_cameras --headless

# python scripts/imitation_learning/Pi0.5/play.py --task Pick_CupWithouthandle_Pour-v0 \
#     --server_host 10.184.17.177 --server_port 8008 \
#     --prompt "pick up the cup without a handle, place it over the yellow bowl and tilt it. " \
#     --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pour/pour_papercup.yaml \
#     --horizon 250 --num_rollouts 30 --enable_cameras --headless

# python scripts/imitation_learning/Pi0.5/play.py --task Pick_cupwithhandle_Pour-v0 \
#     --server_host 10.184.17.177 --server_port 8008 \
#     --prompt "pick up the red mug , place it over the yellow bowl and tilt it. " \
#     --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pour/pour_mugcup.yaml \
#     --horizon 250 --num_rollouts 30 --enable_cameras --headless

# python scripts/imitation_learning/Pi0.5/play.py --task Pick_Stack_Cube-v0 \
#     --server_host 10.184.17.177 --server_port 8008 \
#     --prompt "Grab the red block and place it on the blue block, then grab the green block and place it on the red block." \
#     --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/stack/stack_cube.yaml \
#     --horizon 250 --num_rollouts 30 --enable_cameras --headless

# python scripts/imitation_learning/Pi0.5/play.py --task Pick_PinkCup_Place_WoodShelf-v0 \
#     --server_host 10.184.17.177 --server_port 8008 \
#     --prompt "pick up the pinkcup , place it on the double shelf. " \
#     --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pick/pick_cup_place_woodshelf.yaml \
#     --horizon 250 --num_rollouts 30 --enable_cameras --headless

# python scripts/imitation_learning/Pi0.5/play.py --task Pick_Lid_Place_Pot-v0 \
#     --server_host 10.184.17.177 --server_port 8007 \
#     --prompt "Grab the lid and put it on the pot. " \
#     --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pick/pick_lid_place_pot.yaml \
#     --horizon 250 --num_rollouts 30 --enable_cameras --headless

# python scripts/imitation_learning/Pi0.5/play.py --task Pick_Open_Microwave-v0 \
#     --server_host 10.184.17.177 --server_port 8008 \
#     --prompt "open the microwave door. " \
#     --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/open_close/open_microwave.yaml \
#     --horizon 250 --num_rollouts 30 --enable_cameras --headless

# python scripts/imitation_learning/Pi0.5/play.py --task pick_banana_place_bowl_joint-v0 \
#     --server_host 10.184.17.177 --server_port 8008 \
#     --prompt "Pick up the banana and put it in the yellow bowl." \
#     --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pick/pick_banana_place_bowl.yaml \
#     --horizon 250 --num_rollouts 30 --enable_cameras --headless

python scripts/imitation_learning/Pi0.5/play.py --task Pick_Pour_WhiteKettle4-v0 \
    --server_host 10.184.17.177 --server_port 8008 \
    --prompt "Grab the kettle and move it over the bowl and pour it. " \
    --task_config source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pour/pour_whitekettle4.yaml \
    --horizon 250 --num_rollouts 30 --enable_cameras --headless

    

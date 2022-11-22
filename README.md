# Nuscenes_images_to_yolo
This script can transffer images/labels from nuimages dataset to the form that yolo requires. 

# Prepare for running
1. Download the Nuscenes dataset, where you can access from https://nuscenes.org/
2. Download the nuscenes dev-kit, where you can access from https://github.com/nutonomy/nuscenes-devkit
3. Download the script from this repo
4. Create the directories in which you want to store the result, eg:
   |_ yolo-dataset
      |_ images
          |_ train
          |_ valid
      |_ labels
          |_ train
          |_ valid
         
# Parameters
1. --nu-root，where you put the nuscenes dataset. 
2. --yl-root, where you put the yolo directories that created above.
3. --data-version, which type of data it is, eg train/val/test/mini. If you're not familiar with the operation, please try mini at first.
4. --data-type, the target directory name in point4 above, eg train/valid.
5. --processor, number of threads to deal with the data, depends on your hardware.

# Hope it is useful to you !!

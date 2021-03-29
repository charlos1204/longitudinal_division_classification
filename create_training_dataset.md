# Create a training dataset

Raw images. Starting with a microscopic image:
1. Transform the image into a black and white image. The most common tools are Photoshop, Gimp or a script wrote by you.
2. Cut bacteria from the black and white image and save into single files (jpg or PNG). This process can be manual or automatized.
3. Place the cut black and white images into two folders longitudinal_division and other_division.
4. Repeat the previous steps with the rest of the microscopic images.

**NOTE:** The current model is trained as binary classfier if you want to add more clases, add the folders you need then you must change the code from binary to multiclass classifier. to change a multiclass modify the following line in the notebook from:<br>
`model_ft.fc = nn.Linear(num_ftrs, 2)` <br>
to<br>
`model_ft.fc = nn.Linear(num_ftrs, number_of_classes)` <br>

## Dataset spliting:
1. The common way of spliting is 20% of the images are for testing, then 80% and 20% for traninig and validation with the remaining images.
2. Create a folder that will contain the three following folders: train, val and test
3. Inside of each folder (train, val and test) create two folders: longitudinal_division and other_division
4. Inside of longitudinal_division folder place the black and white bacteria images (jpg or png) with logitudinal division.<br>
![longitudinal1](imgs/longdiv_train_55.jpg) ![longitudinal2](imgs/longdiv_train_72.jpg)
5. Inside of other_division folder place the black and white bacteria images (jpg or png) with other type of division.<br>
![other1](imgs/other_1.jpg) ![other2](imgs/other_2.jpg)

The structure of the folder should be like this:<br>

folder:
  * train:<br>
    * longitudinal_division<br>
    * other_division<br>
  * val:<br>
    * longitudinal_division<br>
    * other_division<br>
  * test:<br>
    * longitudinal_division<br>
    * other_division<br>

<table>
  <thead>
    <tr>
      <th colspan=4>train</th>
      <th colspan=4>val</th>
      <th colspan=4>test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan=2>longitudinal_division</td>
      <td colspan=2>other_division</td>
      <td colspan=2>longitudinal_division</td>
      <td colspan=2>other_division</td>
      <td colspan=2>longitudinal_division</td>
      <td colspan=2>other_division</td>
    </tr>
      <td colspan=2><img src="imgs/train_long_1.jpg" alt=""></img> <img src="imgs/train_long_2.jpg" alt=""></img> <img src="imgs/train_long_3.jpg" alt=""></img>
      <td colspan=2><img src="imgs/train_other_1.jpg" alt=""></img> <img src="imgs/train_other_2.jpg" alt=""></img> <img src="imgs/train_other_3.jpg" alt=""></img>
      <td colspan=2><img src="imgs/val_long_1.jpg" alt=""></img> <img src="imgs/val_long_2.jpg" alt=""></img> <img src="imgs/val_long_3.jpg" alt=""></img>
      <td colspan=2><img src="imgs/val_other_1.jpg" alt=""></img> <img src="imgs/val_other_2.jpg" alt=""></img> <img src="imgs/val_other_3.jpg" alt=""></img>
      <td colspan=2><img src="imgs/test_long_1.jpg" alt=""></img> <img src="imgs/test_long_2.jpg" alt=""></img> <img src="imgs/test_long_3.jpg" alt=""></img>
      <td colspan=2><img src="imgs/test_other_1.jpg" alt=""></img> <img src="imgs/test_other_2.jpg" alt=""></img> <img src="imgs/test_other_3.jpg" alt=""></img>
    <tr>
    </tr>
  </tbody>
</table>

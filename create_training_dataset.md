# Create a training dataset

## Steps:
1. Create a folder that will contain the three following folders: train, val and test
2. Inside of each folder (train, val and test) create the two following folders: longitudinal_division and other_division
3. Inside of longitudinal_division folder place black and white bacteria images (jpg or png) with logitudinal division.<br>
![longitudinal1](imgs/longdiv_train_55.jpg) ![longitudinal2](imgs/longdiv_train_72.jpg)
4. Inside of other_division folder place black and white bacteria images (jpg or png) with other type of division.<br>
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

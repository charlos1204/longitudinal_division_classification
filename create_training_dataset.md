# Create a training dataset

## Steps:
1. Create a folder that will contain the three following folders: train, val and test
2. Inside of each folder (train, val and test) create the two following folders: longitudinal_division and other_division
3. Inside of longitudinal_division folder place black and white bacteria images (jpg or png) with logitudinal division.<br>
![longitudinal1](longdiv_train_55.jpg) ![longitudinal2](longdiv_train_72.jpg)
4. Inside of other_division folder place black and white bacteria images (jpg or png) with other type of division.<br>
![other1](other_1.jpg) ![other2](other_2.jpg)

The structure of the folder should be like this:<br>

<table>
  <thead>
    <tr>
      <th>data_folder</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan=3>train</td>
      <td colspan=3>val</td>
      <td colspan=3>test</td>
    </tr>
  </tbody>
</table>

------ | ------ | ------
train | val | test
------ | ------ | ------
longitudinal_division | other_division | longitudinal_division | other_division | longitudinal_division | other_division 

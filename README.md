# PD_UPDRS_portal

Guide to running the portal

1. Check env version:
``` python3 -m pip --version ```
If you don't have this env then you can install it using:
https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/

2. Create virtual environment:
  
``` python3 -m venv bsn-rating ```

3. Activate virtual environment:
  
```source bsn-rating/bin/activate```

4.  Install package requirements:
5.  
``` pip3 install -r requirements.txt```

1. Change directories to the root folder of the portal

Terminal should look somethihg like this-

```>>(bsn-rating) shehjarsadhu@253 UPDRS_Portal % ```

6. Change dataset path in the code 
In the ```raw_data_routes.py ``` file change the following lines of code at line # 23,24,25.

file_path_root = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_IOTEX/iotex-glove/PD/" to "your file path"
df_dates_file_path = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_IOTEX/iotex-glove/pd_dates_list.csv" "to your file path" 
file_path_filenames = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_IOTEX/iotex-glove/file_paths.csv" to "your_file_path"

7. Run portal:

``` python3 carehub.py```
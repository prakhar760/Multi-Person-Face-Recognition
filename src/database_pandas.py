''' This module contain code to store inferred faces in a pandas dataframe

    The dataframe will have the following columns:
    1. name
    2. id
    3. Time
    4. Date
    5. Cam Name
    6. Cam IP
    7. Face Distance - the distance between the face and the known face (which can be considered
    as inverse of confidence)

    The maximum number of rows in the dataframe will be equal to 1 million

    The dataframe will be stored in a csv file

Author: Anubhav
Date: 24/02/2023
'''

from random import randint
import time
import pandas as pd

from parameters import REPORT_PATH
from custom_logging import logger
from util.generic_utilities import check_for_directory

df_inferred_faces = pd.DataFrame(
    columns=['name', 'id', 'time', 'date', 'cam name', 'cam ip', 'face distance'])


def store_inferred_face_in_dataframe(name_of_person: str, face_distance: float, cam_name: str = 'unknown', cam_ip: str = 'unknown'):
    '''
        This function will store the name of the person and the face distance in the dataframe

        Arguments:
            name_of_person {string} -- name of the person
            face_distance {float} -- face distance
            cam_name {string} -- name of the camera
            cam_ip {string} -- ip address of the camera

        Returns:
            None
    '''

    # current time in HH:MM:SS format
    current_time = time.strftime("%H:%M:%S")
    # current date in DD/MM/YYYY format
    current_date = time.strftime("%d/%m/%Y")

    # The name of person is in the format <name>_<id>
    # Split name_of_person into name and id if there is an underscore in the name
    if '_' in name_of_person:
        name, id = name_of_person.rsplit('_', maxsplit=1)
    else:
        name = name_of_person
        # Give a random id to the person
        id = f'unknown_{randint(0, 1000000)}'

    # We want to store the details of the person only if it is not already present in the dataframe
    if id not in df_inferred_faces['id'].values or \
            (id in df_inferred_faces['id'].values and (name not in df_inferred_faces['name'].values)):
        # store the name of the person in the dataframe
        df_inferred_faces.loc[len(df_inferred_faces)] = [name, id, current_time, current_date, cam_name, cam_ip, face_distance]
        
        # Uncomment the following line to display the details of the person recognized on console
        print(f'Inferred face: {name} ID: {id} from camera: {cam_name} at IP: {cam_ip} at time: {current_time} on date: {current_date}')


def store_dataframe_in_csv():
    '''
        This function will store the dataframe in a csv file

        Arguments:
            None

        Returns:
            True if the dataframe is stored in the csv file, False otherwise
    '''
    global df_inferred_faces

    try:

        # remove the last element from the string as it is the file name
        report_path_dir = REPORT_PATH.rsplit('/', maxsplit=1)[0]
        # create a directory for logs if it does not exist
        check_for_directory(report_path_dir)

        df_inferred_faces.to_csv(REPORT_PATH, encoding='utf-8')
        logger.info('Dataframe stored in csv file')
        return True
    except Exception as e:
        logger.error(
            f'Exception occured while storing dataframe in csv file: {e}')
        return False
    finally:
        # reset the dataframe
        # df_inferred_faces = pd.DataFrame(columns=['name', 'id', 'time', 'date', 'cam name', 'cam ip', 'face distance'])
        pass


def purge_dataframe():
    '''
        This function will purge the dataframe

        Arguments:
            None

        Returns:
            None
    '''
    global df_inferred_faces

    # delete the dataframe
    del df_inferred_faces

    # reset the dataframe
    df_inferred_faces = pd.DataFrame(
        columns=['name', 'id', 'time', 'date', 'cam name', 'cam ip', 'face distance'])

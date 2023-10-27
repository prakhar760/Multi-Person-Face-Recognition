''' This module decouples face recognition logic from rest of the application'''


from parameters import NUMBER_OF_TIMES_TO_UPSAMPLE, BATCH_SIZE, FACE_RECOGNITION_MODEL


def batched_face_detection(batch_of_frames : list):
    ''' This function takes a batch of frames and returns face locations for each frame in the batch
    
    Arguments:
        batch_of_frames {list} -- list of frames
        
    Returns:
        list -- list of face locations for each frame in the batch
    '''

    # Customize following lines of code to get face locations from a batch of frames
    
    # import required libraries
    import face_recognition

    # get the batch of face locations
    batch_of_face_locations = face_recognition.batch_face_locations(batch_of_frames,
                                                                   number_of_times_to_upsample=NUMBER_OF_TIMES_TO_UPSAMPLE,
                                                                   batch_size=BATCH_SIZE)

    # return the batch of face locations
    return batch_of_face_locations


def face_detection(frame : list):
    ''' This function takes a frame and returns face locations for each face in the frame
    
    Arguments:
        frame {list} -- list of frames
        
    Returns:
        list -- list of face locations for each face in the frame
    '''

    # Customize following lines of code to get face locations from a frame

    # import required libraries
    import face_recognition

    # get the face locations
    face_locations_single_frame = face_recognition.face_locations(frame,
                                                        number_of_times_to_upsample=NUMBER_OF_TIMES_TO_UPSAMPLE,
                                                        model=FACE_RECOGNITION_MODEL)

    # return the face locations
    return face_locations_single_frame


def face_embedding(frame, face_locations=None):
    ''' This function takes a frame containing one or more faces and returns face encodings for each face in the frame
    
    Arguments:
        frame {np.array} -- a frame
        
    Returns:
        list of {np.array} -- list of face encodings for each face in the frame
    '''

    # Customize following lines of code to get face encodings from a frame

    # import required libraries
    import face_recognition
    from parameters import FACE_EMBEDDING_MODEL, NUMBER_OF_JITTERS

    # get the face encodings
    face_encodings_single_frame = face_recognition.face_encodings(frame, 
                                                                  face_locations,
                                                                  FACE_EMBEDDING_MODEL,
                                                                  NUMBER_OF_JITTERS
                                                                  )

    # return the face encodings of all faces in the frame
    return face_encodings_single_frame


def face_distance(known_face_encodings, current_face_encoding):
    ''' This function takes a list of face encodings and a single face encoding and returns the distance between each face encoding and the single face encoding
    
    Arguments:
        known_face_encodings {list of {np.array}} -- a list of face encodings
        current_face_encoding {np.array} -- a single face encoding
        
    Returns:
        list of {float} -- list of distances between each face encoding and the single face encoding
    '''

    # Customize following lines of code to get face distances from a list of face encodings and a single face encoding

    # import required libraries
    import face_recognition

    # get the face distances
    face_distances = face_recognition.face_distance(known_face_encodings, current_face_encoding)

    # return the face distances
    return face_distances

import os
import shutil
import random
import time

# DATASET="merged_metric_dataset"
DATA_ROOT="data/SC_sequences"
NEW_DATASET="data/custom_market_dataset_unique"
TRAIN_TEST_SPLIT=0.6
QUERY_PROBABILITY=0.01
TAKE_EVERY=6

TRAIN_DIR_NAME="bounding_box_train"
TEST_DIR_NAME="bounding_box_test"
QUERY_DIR_NAME="query"

def prepare_folders():
    start_fnc = time.time()
    try:
        print("Removing old dataset...")
        shutil.rmtree(os.path.join(
            NEW_DATASET
        ))
        elapsed = time.time() - start_fnc
        print("\tDone in {:.2f} s".format(elapsed))
    except OSError:
        print("\tNot Found")
        pass

    start_new_dirs = time.time()
    print("Creating new dirs...")
    os.makedirs(os.path.join(
        NEW_DATASET,
        TEST_DIR_NAME,
    ))
    os.makedirs(os.path.join(
        NEW_DATASET,
        TRAIN_DIR_NAME,
    ))
    os.makedirs(os.path.join(
        NEW_DATASET,
        QUERY_DIR_NAME,
    ))
    elapsed = time.time() - start_new_dirs
    print("\tDone in {:.2f} s".format(elapsed))

def merge_datasets_to_market():
    start_fnc = time.time()

    train_img_count = 0
    test_img_count = 0
    query_img_count = 0

    id_idx=0

    train_set_count = 0
    test_set_count = 0

    sequence_num = 1

    for sequence_dir in os.listdir(DATA_ROOT):
        start_seq = time.time()
        sequence_path = os.path.join(
            DATA_ROOT,
            sequence_dir,
            "metric_dataset"
        )

        print("Sequence '{:s}':".format(sequence_dir))
        seq_id_idx=0

        for id_dir in os.listdir(sequence_path):
            img_idx=0
            id_dir_full = os.path.join(
                sequence_path,
                id_dir
            )

            train_set = random.random() <= TRAIN_TEST_SPLIT 
            
            if train_set:
                train_set_count += 1
            else:
                test_set_count += 1

            for camera_dir in os.listdir(id_dir_full):
                camera_dir_full = os.path.join(
                    id_dir_full,
                    camera_dir
                )
                camera_id = camera_dir[-1].upper()

                cameras = ["E", "N", "S", "T", "W"]
                camera_num = cameras.index(camera_id)+1

                id_img_i = 0
                for id_img in os.listdir(camera_dir_full):
                    
                    # print(id_img, id_img_i, TAKE_EVERY, id_img_i%TAKE_EVERY)
                    if id_img_i%TAKE_EVERY != 0:
                        id_img_i += 1
                        continue

                    id_img_i += 1

                    query = (not train_set) and (random.random() <= QUERY_PROBABILITY)
                    
                    ext = id_img.split(".")[-1]

                    curr_img_path = os.path.join(
                        camera_dir_full,
                        id_img
                    )

                    new_img_name = "{:04d}_c{:d}s{:d}_{:05d}_{:04d}.{:s}".format(
                        id_idx,
                        camera_num,
                        sequence_num,
                        img_idx,
                        id_idx,
                        ext,
                    )

                    if query:
                        dst_folder = QUERY_DIR_NAME
                        query_img_count += 1
                    elif train_set:
                        dst_folder = TRAIN_DIR_NAME
                        train_img_count += 1
                    else:
                        dst_folder = TEST_DIR_NAME
                        test_img_count += 1
                    
                    new_img_path = os.path.join(
                        NEW_DATASET,
                        dst_folder,
                        new_img_name
                    )

                    shutil.copyfile(
                        curr_img_path,
                        new_img_path
                    )

                    img_idx += 1

            elapsed_total = time.time() - start_fnc
            elapsed = time.time() - start_seq
            print("\r\tIdentity number {:d} (total {:d}) done in {:.2f} s ({:.2f} s total)".format(
                seq_id_idx,
                id_idx,
                elapsed,
                elapsed_total,
            ), end="")
            
            seq_id_idx += 1
            id_idx += 1

        print()
        elapsed = time.time() - start_seq
        print("\tDone in {:.2f} s".format(elapsed))

        sequence_num += 1
    
    elapsed = time.time() - start_fnc
    print("Total transformation time: {:.2f} s".format(elapsed))
        
    print("Number of images for training: {:d}".format(train_img_count))
    print("Number of images for testing: {:d}".format(test_img_count))
    print("Number of images for query: {:d}".format(query_img_count))

    return train_set_count, test_set_count


if __name__ == '__main__':
    prepare_folders()
    train_count, test_count = merge_datasets_to_market()

    print("Number of train IDs: {:d}".format(train_count))
    print("Number of test IDs: {:d}".format(test_count))
    print("Percentage of the train set: {:.2f}%".format(
        train_count/(train_count+test_count) * 100
    ))

import cv2, json, os

def get_img_shape(path):
    img = cv2.imread(path)
    try:
        return img.shape
    except AttributeError:
        print("error! ", path)
        return (None, None, None)

def sorting(l1, l2):
    if l1 > l2:
        lmax, lmin = l1, l2
        return lmax, lmin
    else:
        lmax, lmin = l2, l1
        return lmax, lmin

def convert_labels(path, x1, y1, x2, y2):
    # Definition: Parses label files to extract label and bounding box
    # coordinates. Converts (x1, y1, x1, y2) KITTI format to
    # (x, y, width, height) normalized YOLO format.
    #

    size = get_img_shape(path)
    xmax,xmin = sorting(x1, x2)
    ymax,ymin = sorting(y1, y2)
    dw = 1./size[1]
    dh = 1./size[0]
    x = (xmin + xmax)/2.0
    y = (ymin + ymax)/2.0
    w = xmax - xmin
    h = ymax - ymin
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


def c2y(data_json, img_path,  outpath):

    f=open(data_json)
    training_data = json.load(f)
    check_set = set()
    for i in range(len(training_data['annotations'])):
        image_id = str(training_data['annotations'][i]['image_id'])
        category_id = str(training_data['annotations'][i]['category_id'])
        bbox = training_data['annotations'][i]['bbox']
        image_path = image_id + ".jpg" #TODO
        image_path = os.path.join(img_path, image_path)
        kitti_bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
        yolo_bbox = convert_labels(image_path, kitti_bbox[0], kitti_bbox[1], kitti_bbox[2], kitti_bbox[3])
        filename = image_id + ".txt" #TODO
        content = category_id + " " + str(yolo_bbox[0]) + " " + str(yolo_bbox[1]) + " " + str(yolo_bbox[2]) + " " + str(yolo_bbox[3])
        if image_id in check_set:
            # Append to file
            file = open(filename, "a")
            file.write("\n")
            file.write(content)
            file.close()
        elif image_id not in check_set:
            check_set.add(image_id)
            # Write files
            file = open(filename, "w")
            file.write(content)
            file.close()

def coco2yolo_many_folders(train_inputs,test_inputs, val_inputs, yaml_output):
    pass


if __name__ == "__main__":
    data_json = "/home/borisef/datasets/racoon/train/_annotations.coco.json"
    img_path = "/home/borisef/datasets/racoon/train"
    outpath = "/home/borisef/datasets/outpath/" # will create
    ann_out_dir = 'ann'
    #c2y(data_json=data_json, img_path = img_path, outpath=outpath)

    from pylabel import importer

    dataset = importer.ImportCoco(path=data_json, path_to_images=img_path)

    if(not os.path.exists(outpath)):
        os.mkdir(outpath)
    dataset.export.ExportToYoloV5(output_path=os.path.join(outpath,ann_out_dir),
                                  yaml_file="dataset.yaml",
                                  copy_images=True,
                                  use_splits=False,
                                  cat_id_index=None)

    # need to create folders:
    #outpath/images/train
    #outpath/images/test
    #outpath/images/val

    #outpath/labels/train
    #outpath/labels/test
    #outpath/labels/val

    #yaml
    # train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]


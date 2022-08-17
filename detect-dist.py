# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

# edited
# Distance constants 
KNOWN_DISTANCE = [70.0,18.0,22.5,51.0,7.0,11.0,17.0,35.0,8.5,22,11.0,44.0,10.0,41.5,16.0,9.5] #INCHES
WIDTH = [34.0,3.5,2.0,22.0,3.0,3.0,11.0,14.5,2.5,3.0,11.5,71.0,4.5,60.0,11.0,2.5] #INCHES
width_in_px = [412,181,85,320,250,170,390,1050,195,120,620,530,268,475,384,190] #pixels

def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length

# distance finder function 
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance

@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    ################################################# 1.  Initialize configuration  #####################################################
    #  The path entered becomes a string 
    source = str(source)
    #  Whether to save pictures and txt file 
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    #  Determine whether the file is a video stream 
    # Path() Extract filename   for example ：Path("./data/test_images/bus.jpg") Path.name->bus.jpg Path.parent->./data/test_images Path.suffix->.jpg
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS) #  Extract whether the file suffix meets the requirements , for example ： Whether the format is jpg, png, asf, avi etc. 
    # .lower() Convert to lowercase  .upper() Convert to uppercase  .title() Convert first character to uppercase , The rest are in lowercase , .startswith('http://') return True or Flase
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    # .isnumeric() Whether it is composed of numbers , return True or False
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        #  Return file 
        source = check_file(source)  # download

    # Directories
    #  Predict whether the path exists , There is no new building , According to the experimental documents, a new 
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    #  Get the device  CPU/CUDA
    device = select_device(device)
    #  Detect compilation framework PYTORCH/TENSORFLOW/TENSORRT
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    #  Make sure to enter the size of the picture imgsz aliquot stride=32  If not, adjust to be divisible and return 
    imgsz = check_img_size(imgsz, s=stride)  # check image size
################################################# 2.  Load data  #####################################################
    # Dataloader  Load data 
    #  Use video streams or pages 
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        #  Directly from source Read the picture under the file 
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    #  Saved path 
    vid_path, vid_writer = [None] * bs, [None] * bs
    ############## editing starts ##############
    focal =[]
    for i in range(len(names)):
        focal.append(focal_length_finder(KNOWN_DISTANCE[i], WIDTH[i], width_in_px[i]))  
    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        #  Convert to GPU On 
        im = torch.from_numpy(im).to(device)
        #  Whether to use half precision 
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            #  Add a dimension 
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        #  But the path of the file 
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        # """ pred.shape=(1, num_boxes, 5+num_class) h,w The length and width of the incoming network picture , Be careful dataset Rectangular reasoning is used in detection , So here h Not necessarily equal to w num_boxes = h/32 * w/32 + h/16 * w/16 + h/8 * w/8 pred[..., 0:4] Is the coordinate of the prediction frame = The coordinates of the prediction frame are xywh( Center point + Width length ) Format  pred[..., 4] by objectness Degree of confidence  pred[..., 5:-1] For classification results  """
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        #  Predicted time 
        dt[1] += t3 - t2

        # NMS
        #  Non maximum suppression 
        # """ pred:  The output of the network  conf_thres: Confidence threshold  ou_thres:iou threshold  classes:  Whether to keep only specific categories  agnostic_nms:  Conduct nms Whether to also remove the box between different categories  max-det:  The maximum number of detection frames reserved  ---NMS,  Forecast box format : xywh( Center point + Length and width )-->xyxy( Top left, bottom right ) pred It's a list list[torch.tensor],  The length is batch_size  every last torch.tensor Of shape by (num_boxes, 6),  The content is box + conf + cls """
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        #  Process each picture 
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                #  If the input source is webcam be batch_size>=1  Take out dataset A picture in 
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                #  But most of us usually start from LoadImages Stream to read photos or videos in this file   therefore batch_size=1
                # p:  Current picture / The absolute path of the video   Such as  F:\yolo_v5\yolov5-U\data\images\bus.jpg
                # s:  Output information   For the initial  ''
                # im0:  Original picture  letterbox + pad  Previous pictures 
                # frame:  Video streaming 
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            #  current path yolov5/data/images/
            p = Path(p)  # to Path
            #  picture / The path to save the video save_path  Such as  runs\\detect\\exp8\\bus.jpg
            save_path = str(save_dir / p.name)  # im.jpg
            #  Set the time to save the coordinates of the box txt File path , Each picture corresponds to a frame coordinate information 
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            #  Set the information for printing pictures 
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            #  Save the screenshot 
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                #  Map prediction information to original map 
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                #  Print the number of categories detected 
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                #  Save results ： txt/ Picture frame /crop-image
                for *xyxy, conf, cls in reversed(det):
                    #  The prediction information of each picture is stored in save_dir/labels Under the xxx.txt in   Each row : class_id + score + xywh
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    # #  Draw a frame on the original picture  +  Cut out the predicted target   Save as a picture   Save in save_dir/crops Next   Draw a picture in the original image or save the result 
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        x_min = int(xyxy[0].item())
                        y_min = int(xyxy[1].item())
                        x_max = int(xyxy[2].item())
                        y_max = int(xyxy[3].item())
                        print('bounding box is ', x_min, y_min, x_max, y_max)
                        print('width is ', x_max - x_min)
                        distance = distance_finder (focal[c], WIDTH[c], x_max - x_min)
                        print('distance = ',distance)
                        
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f} {distance:.2f} in')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            #  Draw a frame on the original picture  +  Cut out the predicted target   Save as a picture   Save in save_dir/crops Next 
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            #  display picture 
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            #  Save the picture 
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

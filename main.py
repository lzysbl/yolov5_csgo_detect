#此处为主函数，用于调用模型进行检测当前屏幕，检测到目标后，调用鼠标移动函数进行鼠标移动

import cv2
import torch
import numpy as np
import pyautogui
import time
from pynput import keyboard
from models.common import DetectMultiBackend
from ultralytics.utils.plotting import Annotator, colors, save_one_box
import argparse
import csv
import os
import platform
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode

running = True
stop = True
def on_press(key):
    global running
    global stop
    try:
        if key.char == 'e':
            if(stop):
                stop = False
                print("检测到按键 'e',开始")
                
            else:
                stop = True
                print("检测到按键 'e',暂停")
              # 改变标志变量的值
        if key.char == 'q':
            print("检测到按键 'q'，即将退出...")
            running = False  # 改变标志变量的值
    except AttributeError:
        pass


def run(
    weights=ROOT / "best.pt",  # 模型路径或 Triton URL
    source=ROOT / "screen",  # 文件/目录/URL/glob/screen/0(摄像头)
    data=ROOT / "/home/yaogcc/桌面/py/py应用/鼠标/csgo2_322/data.yaml",  # dataset.yaml 路径
    imgsz=(640, 640),  # 推理尺寸（度，宽度）
    conf_thres=0.3,  # 置信度阈值
    iou_thres=0.45,  # NMS IOU 阈值
    max_det=100,  # 每张图片的最大检测数
    device="",  # CUDA 设备，例如 0 或 0,1,2,3 或 cpu
    view_img=False,  # 显示结果
    save_txt=False,  # 保存结果为 *.txt
    save_csv=False,  # 以 CSV 格式保存结果
    save_conf=False,  # 在 --save-txt 标签中保存置信度
    save_crop=False,  # 保存裁剪后的预测框
    nosave=False,  # 不保存图像/视频
    classes=None,  # 按类别过滤：--class 0，或 --class 0 2 3
    agnostic_nms=False,  # 类别不敏感的 NMS
    augment=False,  # 增强推理
    visualize=False,  # 可视化特征
    update=False,  # 更新所有模型
    project=ROOT / "runs/detect",  # 将结果保存到 project/name
    name="exp",  # 将结果保存到 project/name
    exist_ok=False,  # 项目/名称已存在时不增加后缀
    line_thickness=1,  # 边界框厚度（像素）
    hide_labels=False,  # 隐藏标签
    hide_conf=False,  # 隐藏置信度
    half=False,  # 使用 FP16 半精度推理
    dnn=False,  # 使用 OpenCV DNN 进行 ONNX 推理
    vid_stride=1,  # 视频帧率步长
): 
    pyautogui.FAILSAFE = False
    global running
    is_move = 3
    source = str(source)  # 将源路径转换为字符串类型
    save_img = not nosave and not source.endswith(".txt")  # 保存推理结果图像，如果未禁止保存且不是以".txt"结尾的话
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)  # 检查源是否为文件（图像或视频）
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))  # 检查源是否是URL
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)  # 检查是否是摄像头、streams文件或URL
    screenshot = source.lower().startswith("screen")  # 检查是否是截屏

    if is_url and is_file:
        source = check_file(source)  # 如果是URL并且是文件，下载文件

    # 目录相关操作
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # 递增运行目录
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 创建目录，如果保存标签则创建 "labels" 子目录

    # 加载模型
    device = select_device(device)  # 选择设备（GPU或CPU）
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)  # 加载模型
    stride, names, pt = model.stride, model.names, model.pt  # 模型步长、类别名称、模型

    imgsz = check_img_size(imgsz, s=stride)  # 检查图像大小

    # 数据加载器
    bs = 1  # 批量大小
    if webcam:
        view_img = check_imshow(warn=True)  # 检查是否可以显示图像
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)  # 如果是摄像头，将批量大小设置为数据集长度
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)  # 如果是截屏，加载截屏数据集
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)  # 加载图像数据集

    vid_path, vid_writer = [None] * bs, [None] * bs  # 初始化视频路径和视频写入器列表


    # 进行推理
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # 预热
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        if not running:  # 如果 running 为 False，则退出循环
            break
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 转为 fp16/32
            im /= 255  # 0 - 255 转为 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # 扩展为批次维度
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # 推理
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        # 非极大值抑制
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # 第二阶段分类器（可选）
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # 定义CSV文件路径
        csv_path = save_dir / "predictions.csv"

        # 创建或追加到CSV文件
        def write_to_csv(image_name, prediction, confidence):
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # 处理预测结果
        for i, det in enumerate(pred):  # 每张图片
            if running == False:
                break
            
            seen += 1
            if webcam:  # 批量大小 >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)


            p = Path(p)  # 转换为 Path 对象
            save_path = str(save_dir / p.name)  # 图像保存路径，例如 im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # 标签保存路径，例如 im.txt
            s += "%gx%g " % im.shape[2:]  # 打印字符串，显示图像尺寸
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 归一化增益 whwh
            imc = im0.copy() if save_crop else im0  # 用于保存裁剪图像
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))  # 创建图像注释器
            if len(det):
                is_move+=1
                # 将边界框从 img_size 缩放到 im0 尺寸
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # 打印检测结果
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # 每个类别的检测数量
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 添加到字符串中
                #获取检测框的中心点坐标
                
                
                if stop==False :
                    # 获取目标的边界框坐标
                    x1, y1, x2, y2 = det[:4]
                    # 计算目标的中心点坐标
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2
                    # 计算目标中心与鼠标位置的距离
                    mouse_pos_x, mouse_pos_y = pyautogui.position()
                    
                    print(f"当前鼠标指针的位置是：({mouse_pos_x}, {mouse_pos_y})")
                    print(f"检测到目标，目标位置：({x_center}, {y_center})")
                    pyautogui.moveTo(x_center, y_center)
                    pyautogui.click()
                    print(f"点击鼠标左键")
                    


                # 写入结果
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # 类别的整数表示
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    if save_csv:  # 写入CSV文件
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # 写入文本文件
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # 归一化的 xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # 标签格式
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # 添加边界框到图像
                        c = int(cls)  # 类别的整数表示
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:  # 保存裁剪图像
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)
 
            # Stream results (流式展示结果)
            im0 = annotator.result()  # 获取注释后的图像
            if view_img:  # 如果设置为显示图像
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # 允许窗口调整大小 (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])  # 调整窗口大小以适应图像
                cv2.imshow(str(p), im0)  # 显示图像
                cv2.waitKey(1)  # 等待1毫秒

            # Save results (保存结果，包括带有检测的图像)

            if save_img:  # 如果设置为保存图像
                if dataset.mode == "image":  # 如果处理的是单张图像
                    cv2.imwrite(save_path, im0)  # 保存注释后的图像
                else:  # 如果处理的是 'video' 或 'stream'
                    if vid_path[i] != save_path:  # 新的视频文件
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # 释放之前的视频写入器资源
                        if vid_cap:  # 处理视频文件
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频帧的宽度
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频帧的高度
                        else:  # 处理流式输入
                            fps, w, h = 30, im0.shape[1], im0.shape[0]  # 默认帧率、图像宽度和高度
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # 强制结果视频文件使用 *.mp4 后缀
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))  # 创建视频写入器并打开文件
                    vid_writer[i].write(im0)  # 将注释后的图像帧写入视频文件


        # 打印推理时间（仅限推理部分）
    LOGGER.info(f"{s}{'' if len(det) else '(未检测到物体), '}{dt[1].dt * 1E3:.1f}ms")

    # 打印结果
    t = tuple(x.t / seen * 1e3 for x in dt)  # 每张图片的速度
    LOGGER.info(f"速度: %.1fms 预处理, %.1fms 推理, %.1fms NMS 每张图片大小为 {(1, 3, *imgsz)}" % t)

    if save_txt or save_img:  # 如果要保存文本或图像结果
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} 个标签已保存到 {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"结果已保存到 {colorstr('bold', save_dir)}{s}")  # 打印结果保存信息

    if update:  # 如果要更新模型
        strip_optimizer(weights[0])  # 更新模型（以修复 SourceChangeWarning）


if __name__ == "__main__":
    keyboard_listener = keyboard.Listener(on_press=on_press)
    keyboard_listener.start()
    # 示例用法
    run()  # 运行推理

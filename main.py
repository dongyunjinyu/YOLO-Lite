# YOLO-Lite 🚀

from yololite import YOLOLite

# 加载预训练模型
model = YOLOLite("yolo11n.pt")

# 不使用预训练模型，会导致损失难以下降
# model = YOLOLite("yololite3d/cfg/yolo11.yaml")

# 训练coco8
results = model.train(data="coco8/coco8.yaml", epochs=1, imgsz=640)

# 推理
# results = model(["boats.jpg"])
# print(results[0].boxes)

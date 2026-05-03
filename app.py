import gradio as gr
from pathlib import Path
from ultralytics import YOLOv10

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "runs/puddle/yolov10n_puddle_improved/weights/best.pt"
APP_CSS = """
    .gradio-container {
        font-size: 18px !important;
    }
    .main-title {
        text-align: center;
        font-size: 34px;
        font-weight: 700;
        margin: 10px 0 4px;
    }
    .sub-title {
        text-align: center;
        font-size: 22px;
        font-weight: 500;
        margin: 0 0 18px;
        color: #444;
    }
    label, .block-title, .wrap label {
        font-size: 18px !important;
        font-weight: 600 !important;
    }
    button {
        font-size: 20px !important;
        font-weight: 600 !important;
    }
    input, textarea {
        font-size: 17px !important;
    }
"""


def puddle_detect(image, model_path, image_size, conf_threshold):
    model = YOLOv10(model_path)
    results = model.predict(source=image, imgsz=image_size, conf=conf_threshold)
    annotated_image = results[0].plot()
    return annotated_image[:, :, ::-1]


def app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="输入图片", visible=True)
                model_path = gr.Textbox(
                    label="模型权重路径（best.pt）",
                    value=str(DEFAULT_WEIGHTS),
                )
                image_size = gr.Slider(
                    label="输入图像尺寸",
                    minimum=320,
                    maximum=1280,
                    step=32,
                    value=640,
                )
                conf_threshold = gr.Slider(
                    label="置信度阈值",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.25,
                )
                detect_btn = gr.Button(value="开始检测")

            with gr.Column():
                output_image = gr.Image(type="numpy", label="检测结果图", visible=True)

        detect_btn.click(
            fn=puddle_detect,
            inputs=[image, model_path, image_size, conf_threshold],
            outputs=[output_image],
        )

gradio_app = gr.Blocks(css=APP_CSS)
with gradio_app:
    gr.HTML("<div class='main-title'>雨后路面积水检测系统</div>")
    gr.HTML("<div class='sub-title'>基于改进YOLOv10模型</div>")
    with gr.Row():
        with gr.Column():
            app()
if __name__ == '__main__':
    gradio_app.launch()

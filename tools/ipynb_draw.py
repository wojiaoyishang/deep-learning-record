from ipywidgets import FileUpload, VBox, Image, IntSlider, ColorPicker, Button, HBox, Layout, Output
from ipycanvas import Canvas, hold_canvas


class DrawCanvas:
    def __init__(self, width=300, height=300):
        self.canvas = Canvas(
            width=width,
            height=height,
            sync_image_data=True,
            layout=Layout(width='60%', border='2px solid black')
        )
        self.fill_style = 'black'
        self.canvas.fill_style = self.fill_style  # Set canvas background to white
        self.canvas.fill_rect(0, 0, self.canvas.width, self.canvas.height)

        # Create file uploader
        self.uploader = FileUpload(accept='.png,.jpg,.jpeg', multiple=False)
        self.out = Output()

        # Create drawing tools
        self.brush_size = IntSlider(value=2, min=1, max=20, description='Brush Size:')
        self.color_picker = ColorPicker(description='Color:', value='#ffffff')
        self.eraser_button = Button(description='Eraser', button_style='danger')
        self.clear_button = Button(description='Clear Canvas', button_style='warning')

        # Drawing state
        self.drawing = False
        self.is_eraser = False
        self.last_x, self.last_y = None, None

        # Bind button events
        self.uploader.observe(self.on_file_upload, names='value')

        self.eraser_button.on_click(self.toggle_eraser)
        self.clear_button.on_click(self.clear_canvas)

        # Bind canvas drawing events
        self.canvas.on_mouse_down(self.start_drawing)
        self.canvas.on_mouse_up(self.stop_drawing)
        self.canvas.on_mouse_move(self.draw)

    # File upload handler
    def on_file_upload(self, change):
        self.canvas.clear()
        self.canvas.fill_style = 'white'
        self.canvas.fill_rect(0, 0, self.canvas.width, self.canvas.height)
        uploaded_file = change['new']
        if uploaded_file:
            file_content = uploaded_file[0]['content']
            image = Image(value=file_content)
            self.canvas.draw_image(image, 0, 0, self.canvas.width, self.canvas.height)

    def start_drawing(self, x, y):
        self.drawing = True
        self.last_x, self.last_y = x, y

    def stop_drawing(self, x, y):
        self.drawing = False
        self.last_x, self.last_y = None, None

    def draw(self, x, y):
        if self.drawing and self.last_x is not None and self.last_y is not None:
            with hold_canvas(self.canvas):
                self.canvas.line_width = self.brush_size.value
                self.canvas.stroke_style = 'white' if self.is_eraser else self.color_picker.value
                self.canvas.line_cap = 'round'
                self.canvas.begin_path()
                self.canvas.move_to(self.last_x, self.last_y)
                self.canvas.line_to(x, y)
                self.canvas.stroke()
            self.last_x, self.last_y = x, y

    # Eraser toggle
    def toggle_eraser(self, change):
        self.is_eraser = not self.is_eraser
        self.eraser_button.button_style = 'success' if self.is_eraser else 'danger'
        self.eraser_button.description = 'Brush' if self.is_eraser else 'Eraser'

    # Clear canvas
    def clear_canvas(self, change):
        self.canvas.clear()
        self.canvas.fill_style = self.fill_style
        self.canvas.fill_rect(0, 0, self.canvas.width, self.canvas.height)

    def get_layout(self):
        return VBox([self.uploader,
                     HBox(
                         [self.brush_size, self.color_picker, self.eraser_button, self.clear_button]
                     ),
                     self.canvas,
                     self.out
                     ])

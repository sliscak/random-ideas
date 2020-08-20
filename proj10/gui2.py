import PySimpleGUI as sg
import io
import base64
from PIL import Image
from tkinter import PhotoImage
from time import time

def convert(image):
    # image = Image(image)
    with io.BytesIO() as output:
        image.save(output, format='png')
        b = output.getvalue()
        base64_img = base64.b64encode(b)
    return base64_img
    # image = PhotoImage(image)

layout = []
layout.append(
    [sg.Text('Ahoj PySimpleGUI', key='welcome_text')]
)
layout.append(
    [sg.Graph(
            canvas_size=(400, 400),
            graph_bottom_left=(0, 0),
            graph_top_right=(400, 400),
            background_color='white',
            enable_events= True,
            drag_submits= True,
            key='graph'
        ),
        sg.Button('RELOAD', key='reload', size=(10, 2), pad=((0, 0), (350, 0))),
        sg.Button('ERASE', tooltip='Cancell this action', size=(10, 2), pad=((5, 0), (350, 0)))
    ]
)
layout.append(
    [sg.InputText('image_path', size=(100, 0), key='path_text_box'),
        sg.FileBrowse('Open Image', file_types=(('Images',('*.jpg', '*.png')),), enable_events=True, key='load_img')
     ]
)
layout.append([sg.OK(f'{x}') for x in range(10)])

try:
    window = sg.Window(title='AI E-DIT', layout=layout, margins=(100, 50))
    # window.finalize()
    images = []
    # graph = window['graph']
    graph = window.FindElement('graph')
    while True:
        event, values = window.read(timeout=100)
        w_text = window.FindElement('welcome_text')
        w_text.update(f'{time()}')
        # print(event, values)
        if event is not None:
            print(f'Event: {event}\nValues: {values}')
            if event == sg.WINDOW_CLOSED:
                break
            elif event == 'load_img':
                f_url = values.get('load_img')
                print('here')
                if len(f_url) > 0:
                    image = Image.open(f_url)
                    image = image.resize((400, 400))
                    images= [image]
                    base64_img = convert(image)
                    graph.draw_image(data=base64_img, location=(0, 400))
                path_text_box = window.FindElement('path_text_box')
                path_text_box.Update(f_url)
                print(path_text_box.Update)
                print('Image loaded')
            elif event == 'reload':
                f_url = values.get('load_img')
                if len(f_url) > 0:
                    image = Image.open(f_url)
                    image = image.resize((400, 400))
                    images = [image]
                    base64_img = convert(image)
                    graph.draw_image(data=base64_img, location=(0, 400))

            elif event == 'ERASE':
                print('KUA')
                graph.Erase()
            elif event == 'graph': #or (event == 'graph+UP'):
                # print(dir(images[0]))
                # print(help(images[0].putpixel))
                x,y = values.get('graph')
                images[0].putpixel((x, -y), (0, 0, 0))
                base64_img = convert(images[0])
                print(dir(graph))
                exit()
                graph.draw_image(data=base64_img, location=(0, 400))
                print('prid')
except Exception as e:
    pass
finally:
    window.close()
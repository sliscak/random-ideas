import PySimpleGUI as sg
import io
import base64
from PIL import Image
from tkinter import PhotoImage

def convert(image):
    # image = Image(image)
    with io.BytesIO() as output:
        image.save(output, format='png')
        b = output.getvalue()
        base64_img = base64.b64encode(b)
    return base64_img
    # image = PhotoImage(image)

layout = [
    [sg.Text('Ahoj PySimpleGUI')],
    [sg.Graph(
        canvas_size=(400, 400),
        graph_bottom_left=(0, 0),
        graph_top_right=(400, 400),
        background_color='white',
        enable_events= True,
        drag_submits= True,
        key='graph'
    )],
    [sg.Button('LOAD')],
    [sg.Button('ERASE', tooltip='Cancell this action', size=(10, 2))],
    [sg.FileBrowse('Open Image', file_types=(('Images',('*.jpg', '*.png')),))],
]

try:
    window = sg.Window(title='Ahoj Svet', layout=layout, margins=(100, 50))
    # window.finalize()
    images = []
    # graph = window['graph']
    graph = window.FindElement('graph')
    while True:
        event, values = window.read()
        if event is not None:
            print(f'Event: {event}\nValues: {values}')
            if event == sg.WINDOW_CLOSED:
                break
            elif event == 'LOAD':
                f_url = values.get('Open Image')
                if len(f_url) > 0:
                    print('reading')
                    # graph.draw_image(filename=f_url, location=(0, 400))
                    # image = open(f_url, 'rb')
                    # print(f'f_url = {f_url}')
                    image = Image.open(f_url)#.convert('RGB').tobytes()
                    image = image.resize((400, 400))
                    images.append(image)
                    base64_img = convert(image)
                    # images = [image]
                    # # print(dir(image))
                    # print('reading')
                    # print(graph)
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
import PySimpleGUI as sg
from time import ctime

APP_NAME = 'Mini Editor Alpha'

def init_layout():
    layout = [
        [
            sg.Text('', auto_size_text=False, key='time-text'),
            sg.Multiline('Alfa'),
            sg.Input('Input here')
        ],
        [
            sg.Menu(menu_definition=[
                ['File', ['New::new-file', 'Open::open-file']],
                ['Help', ['About']]
          ])
        ],
        [
            sg.Exit('EXIT', pad=(200, 0,0,0))
        ],
        # [
        #     sg.Menu(menu_definition=[['&File', ['!&Open', '&Save::savekey', '---', '&Properties', 'E&xit']],
        #     ['&Edit', ['!&Paste', ['Special', 'Normal', ], 'Undo'], ],
        #     ['&Debugger', ['Popout', 'Launch Debugger']],
        #     ['&Toolbar', ['Command &1', 'Command &2', 'Command &3', 'Command &4']],
        #     ['&Help', '&About...'], ])
        # ],
    ]
    return layout


if __name__ == '__main__':
    sg.ChangeLookAndFeel('DarkGrey')
    window = sg.Window(title=APP_NAME, layout=init_layout(), margins=(200, 150), return_keyboard_events=True)
    # window.Maximize()
    time_text = window.FindElement('time-text')
    while True:
        event, values = window.read(timeout=250)
        time_text.update(ctime())
        window.set_title(f'{APP_NAME}   {ctime()}')
        if event == sg.WINDOW_CLOSED or event == 'EXIT' or event == sg.WIN_CLOSED:
            break
        elif (event is not None) or (values is not None):
            print(f'E: {event},\tV: {values}')
            if event == 'About':
                sg.popup(APP_NAME)

    window.close()
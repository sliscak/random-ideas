import PySimpleGUI as sg
from time import ctime

APP_NAME = 'Mini Editor Alpha'
SPACE = ''.join(' ' for x in range(5))
EXIT_EVENTS = (sg.WINDOW_CLOSED, sg.WIN_CLOSED, 'EXIT')

def init_layout():
    layout = [
        [
            sg.Menu(menu_definition=[
                ['File', ['New::new-file', 'Open::open-file']],
                ['Help', ['About']],
            ])
        ],
        [
            sg.Multiline('Alfa', size=(80, 5), pad=(0,80,0,0), key='multiline-field', disabled=True),
        ],
        [
            sg.Text('TIMESTAMP\t:', auto_size_text=False, key='input-text1')
        ],
        [
            sg.Text('CONTENT\t:', auto_size_text=False, key='input-text2')
        ],
        [
            # sg.Text('', auto_size_text=False, key='input-text'),
            sg.InputText('', pad=(0, 20, 0,0), size=(80, 0), key='input-field', enable_events=True),
            sg.Exit('EXIT'),
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
    window = sg.Window(title=APP_NAME, layout=init_layout(), return_keyboard_events=True, use_default_focus=False)
    while True:
        event, values = window.read(250)
        if event in EXIT_EVENTS:
            break
        window.set_title(f'{APP_NAME}{SPACE}{ctime()}')
        if (event is not None) or (values is not None):
            # print(f'E: {event},\tV: {values}')
            # if len(event) == 1:
            #     print(f'E: {ord(event)},\tV: {values}')
            # if event == chr(13):
            # if event == 'Control_L:17':
            #     print('CONTROL L17!')
            if event == chr(13):
                timestamp = ctime()
                it1 = window.FindElement('input-text1')
                it1.update(f'TIMESTAMP\t: {timestamp}')

                val = values.get('input-field')
                it2 = window.FindElement('input-text2')
                it2.update(f'CONTENT\t: {val}')
            # if event == chr(13) and event == 'Control_L:17':
            #     print('WHAT')
            elif event == 'input-field':
                inp_field = window.FindElement('input-field')
                val = inp_field.Get()
                if len(val) > 50:
                    inp_field.update(val[0:50])
            elif event == 'About':
                sg.popup(APP_NAME)

    window.close()
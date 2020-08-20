import PySimpleGUI as sg

def init_layout():
    layout = [
        [
            sg.Slider(orientation='horizontal', range=(0, 100), enable_events=True, tick_interval=25)
        ],
        [
          sg.Menu(menu_definition=[
              ['File', ['New::new-file', 'Open::open-file']]
          ])
        ],
        [
            sg.Exit('EXIT')
        ]
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
    window = sg.Window(title='Window', layout=init_layout(), margins=(100, 50))
    while True:
        event, values = window.read()
        if (event is not None) or (values is not None):
            print(f'E: {event},\tV: {values}')
            if event == sg.WINDOW_CLOSED or event == 'EXIT':
                break

    window.close()
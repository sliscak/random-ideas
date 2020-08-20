import PySimpleGUI as sg
import shelve
from time import ctime, time
from datetime import datetime
from random import choices, randint

APP_NAME = 'Mini Reminder Alpha'#'Mini Editor Alpha'
SPACE = ''.join(' ' for x in range(5))
EXIT_EVENTS = (sg.WINDOW_CLOSED, sg.WIN_CLOSED, 'EXIT')
DATABASE_NAME = 'database'
MAX_ITEMS = 100

def init_layout():
    layout = [
        [
            sg.Menu(menu_definition=[
                ['File', ['New::new-file', 'Open::open-file']],
                ['Help', ['About']],
            ])
        ],
        [
            sg.Slider(range=(0, 100), orientation='h', size=(62, 5), default_value=50, tick_interval=25, key='priority-slider2', disabled=True),
        ],
        [
            sg.Multiline('---default---', size=(80, 5), pad=(0,(40,10),0), key='multiline-field', disabled=True),
        ],
        [
            sg.Button('Pause', key='pause-button', size=(10, 1), pad=((450, 0), (0, 40), 0, 50), button_color=('black', 'orange')),
            sg.Button('Delete', key='delete-button', size=(10, 1), pad=((0, 0), (0,40), 0,50), button_color=('white', 'brown')),
        ],
        [
            sg.Text('TIMESTAMP\t:', auto_size_text=False, key='input-text1')
        ],
        [
            sg.Text('CONTENT\t:', auto_size_text=False, key='input-text2'),
        ],
        [
            sg.Text('Priority:'),
            sg.Slider(range=(0, 100), orientation='h', size=(40, 20), default_value=50, tick_interval=25, key='priority-slider'),
            sg.Button('RESET', key='priority-reset-button', size=(10, 1), button_color=('white', 'darkgreen'))
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
    with shelve.open(DATABASE_NAME) as db:
        t0 = time()
        current_key = None
        pause = False
        while True:
            event, values = window.read(250)
            if event in EXIT_EVENTS:
                break
            window.set_title(f'{APP_NAME}{SPACE}{ctime()}')
            print(f'{list(db.keys())}\n{db.values()}')
            t = time()
            if ((t - t0) >= 3) and not pause:
                t0 = t
                if len(db.keys()) > 0:
                    weights = [db[key][2]+1 for key in db.keys()]
                    key = choices(list(db.keys()), weights=weights, k=1)[0]
                    val = db.get(key)
                    current_key = key
                    mlf = window['multiline-field']
                    mlf.update(f'{val[0].strftime("%d-%m-%Y %H:%M:%S")}\n{val[1]}')
                    prs2 = window['priority-slider2']
                    prs2.update(value=val[2])

            if (event is not None) or (values is not None):
                if event == chr(13):
                    timestamp = datetime.utcnow()
                    it1 = window['input-text1']
                    it1.update(f'TIMESTAMP\t: {timestamp.strftime("%d-%m-%Y %H:%M:%S")}')

                    val = values.get('input-field')
                    priority = values.get('priority-slider')
                    it2 = window['input-text2']
                    it2.update(f'CONTENT\t: {val}')
                    # db[str(len(db))] = (timestamp, val, priority)
                    if len(db.keys()) < MAX_ITEMS:
                        r = randint(0, MAX_ITEMS)
                        while db.get(str(r)) is not None:
                            r = randint(0, MAX_ITEMS)
                        db[str(r)] = (timestamp, val, priority)
                        print(len(db))
                    else:
                        raise Exception('Max Number of Items in DATABASE!')

                    # sdr = window['priority-slider']
                    # sdr.update(value=50)
                elif event == 'input-field':
                    inp_field = window['input-field']
                    val = inp_field.Get()
                    # if len(val) > 50:
                    #     inp_field.update(val[0:50])
                elif event == 'delete-button':
                    if current_key is not None:
                        v = db[current_key]
                        del db[current_key]
                        print(f'Deleted: K: {current_key} V: {v}')
                        current_key = None
                        mlf = window['multiline-field']
                        mlf.update('')
                elif event == 'pause-button':
                    # pause = True if pause is False else False
                    pause_button = window['pause-button']
                    if pause is False:
                        pause = True
                        pause_button.update('Continue', button_color=('white', 'green'))
                    else:
                        pause = False
                        pause_button.update('Pause', button_color=('black', 'orange'))
                elif event == 'priority-reset-button':
                    prs = window['priority-slider']
                    prs.update(value=50)

                elif event == 'About':
                    sg.popup(APP_NAME)

        window.close()
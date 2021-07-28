import PySimpleGUI as sg
import shelve
from time import ctime, time
from datetime import datetime
from random import choices, randint
import jsonlines
from collections import namedtuple

APP_NAME = 'Mini Reminder Alpha'  # 'Mini Editor Alpha'
SPACE = ''.join(' ' for x in range(5))
EXIT_EVENTS = (sg.WINDOW_CLOSED, sg.WIN_CLOSED, 'EXIT')
DATABASE_NAME = 'database'
MAX_ITEMS = 100


def init_layout():
    """ Here is the whole layout defined"""
    layout = [
        [
            sg.Menu(menu_definition=[
                ['File', ['New::new-file', 'Open::open-file']],
                ['Help', ['About']],
            ])
        ],
        [
            sg.Slider(range=(0, 100), orientation='h', size=(62, 5), default_value=50, tick_interval=25,
                      key='priority-slider2', disabled=True),
        ],
        [
            sg.Multiline('---default---', size=(80, 5), pad=(0, (40, 10), 0), key='multiline-field', disabled=True),
        ],
        [
            sg.Button('Pause', key='pause-button', size=(10, 1), pad=((450, 0), (0, 10), 0, 50),
                      button_color=('black', 'orange')),
            sg.Button('Delete', key='delete-button', size=(10, 1), pad=((0, 0), (0, 10), 0, 50),
                      button_color=('white', 'brown')),
        ],

        [
            sg.Text('TIMESTAMP\t:', auto_size_text=False, key='input-text1'),
            sg.Button('Done', key='done-button', size=(15, 2), pad=((100, 0), (0, 0), (0, 0)),
                      button_color=('white', 'green')),
        ],
        [
            sg.Text('CONTENT\t:', auto_size_text=False, key='input-text2'),
        ],
        [
            sg.Text('Priority:'),
            sg.Slider(range=(0, 100), orientation='h', size=(40, 20), default_value=50, tick_interval=25,
                      key='priority-slider'),
            sg.Button('RESET', key='priority-reset-button', size=(10, 1), button_color=('white', 'darkgreen'))
        ],
        [
            # sg.Text('', auto_size_text=False, key='input-text'),
            sg.InputText('', pad=(0, 20, 0, 0), size=(80, 0), key='input-field', enable_events=True),
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


# replace status 'ongoin' with 'created'??
# replace generic 'timestamp' with 'created' key.
Task = namedtuple('Task', ['content', 'timestamp', 'priority', 'status', 'updated_on'])

if __name__ == '__main__':
    sg.ChangeLookAndFeel('DarkGrey')
    window = sg.Window(title=APP_NAME, layout=init_layout(), return_keyboard_events=True,
                       use_default_focus=False)  # no context manager x
    with jsonlines.open('database.jsonl', mode='r') as reader:
        try:
            db = {key: Task(**val) for key, val in enumerate(reader.iter())}
        except Exception as e:
            print('Failed reading Database!')
    with jsonlines.open('database_z.jsonl', mode='a', flush=True) as jsonl_writer:
        t0 = time()
        current_key = None
        pause = False
        while True:
            event, values = window.read(250)
            if event in EXIT_EVENTS:
                try:
                    with jsonlines.open('database.jsonl', mode='+') as writer:
                        print(dir(writer))
                        for task in db.values():
                            writer.write(task._asdict())
                except Exception as e:
                    print('Failed writing the Database!')
                break
            window.set_title(f'{APP_NAME}{SPACE}{ctime()}')
            # print(f'{list(db.keys())}\n{db.values()}')
            t = time()
            if ((t - t0) >= 3) and not pause:
                t0 = t
                if len(db.keys()) > 0:
                    weights = [db[key].priority + 1 for key in db.keys()]
                    current_key = choices(list(db.keys()), weights=weights, k=1)[0]
                    task = db.get(current_key)
                    mlf = window['multiline-field']
                    mlf.update(f'{task.timestamp}\n{task.content}')
                    prs2 = window['priority-slider2']
                    prs2.update(value=task.priority)

            if (event is not None) or (values is not None):
                if event == chr(13):
                    """
                        Add a new task after pressing the enter key.
                    """
                    timestamp = datetime.now()
                    timestamp = timestamp.strftime("%d-%m-%Y %H:%M:%S")
                    it1 = window['input-text1']
                    it1.update(f'TIMESTAMP\t: {timestamp}')

                    content = values.get('input-field')
                    priority = values.get('priority-slider')

                    it2 = window['input-text2']
                    it2.update(f'CONTENT\t: {content}')

                    task = Task(content=content, timestamp=timestamp, priority=priority, status='ongoing',
                                updated_on=None)
                    db[str(len(db))] = task
                    jsonl_writer.write(task._asdict())

                elif event == 'input-field':
                    inp_field = window['input-field']
                    val = inp_field.Get()

                elif event == 'delete-button':
                    if current_key is not None:
                        updated_on = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                        print(updated_on)
                        task = db[current_key]
                        task = task._replace(status='deleted', updated_on=updated_on)
                        jsonl_writer.write(task._asdict())
                        print(f'Deleted: K: {current_key} V: {task}')
                        mlf = window['multiline-field']
                        mlf.update('')
                        del db[current_key]
                        current_key = None

                elif event == 'pause-button':
                    # pause = True if pause is False else False
                    pause_button = window['pause-button']
                    if pause is False:
                        pause = True
                        pause_button.update('Continue', button_color=('white', 'green'))
                        prs2 = window['priority-slider2']
                        prs2.update(disabled=False)
                        multiline_field = window['multiline-field']
                        multiline_field.update(disabled=False)
                    else:
                        if current_key is not None:
                            prs2 = window['priority-slider2']
                            multiline_field = window['multiline-field']

                            task = db[current_key]
                            new_priority = values.get('priority-slider2')
                            # new_content = values.get('multiline-field').splitlines()[1] # get the text string afte the first line(because first line is timestamp
                            new_content = values.get('multiline-field').split(maxsplit=2)[2].strip()

                            if new_priority != task.priority or task.content != new_content:
                                updated_on = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                                task = task._replace(content=new_content, priority=new_priority, updated_on=updated_on)
                                db[current_key] = task
                                jsonl_writer.write(task._asdict())
                                print(f'Priority updated: K: {current_key} V: {task}')

                                prs2.update(value=task.priority, disabled=True)
                        pause = False
                        pause_button.update('Pause', button_color=('black', 'orange'))


                elif event == 'done-button':
                    if current_key is not None:
                        updated_on = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                        task = db[current_key]
                        task = task._replace(status='done', updated_on=updated_on)
                        print(f'Done: K: {current_key} V: {task}')
                        mlf = window['multiline-field']
                        mlf.update('')
                        jsonl_writer.write(task._asdict())
                        del db[current_key]
                        current_key = None

                elif event == 'priority-reset-button':
                    prs = window['priority-slider']
                    prs.update(value=50)

                elif event == 'About':
                    sg.popup(APP_NAME)

        window.close()

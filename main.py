import random as rnd
import time, msvcrt, re
import math as mth
import numpy as np
import math_matrix as mth_mtx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import multiprocessing as mp
from scipy.stats import chisquare
from scipy.optimize import minimize
from math_constant import constant

def plot_tournament_results(sorted_result, strategies_category, strategies_creator,
                             tournament_divider, time_divider, top_n=30, show_worst=False):
    data = sorted_result[-top_n:][::-1] if show_worst else sorted_result[:top_n]
    names  = [item[1] for item in data]
    scores = [item[0] / tournament_divider for item in data]
    wins   = [item[2] / time_divider for item in data]
    draws  = [item[3] / time_divider for item in data]
    loses  = [item[4] / time_divider for item in data]
    colors = ['#2ecc71' if strategies_category[n] == 'NICE' else '#e74c3c' for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(names) * 0.32)))
    y_pos = np.arange(len(names))
    title_prefix = 'Worst' if show_worst else 'Best'
    bars_left = axes[0].barh(y_pos, scores, color=colors)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels([])
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Average score per tournament')
    axes[0].set_title(f'{title_prefix} {len(names)} Strategy based on the game_score')
    max_score = max(scores) if scores else 1
    for i, (bar, v, name) in enumerate(zip(bars_left, scores, names)):
        axes[0].text(max_score * 0.01, bar.get_y() + bar.get_height() / 2,
                     name, va='center', ha='left', fontsize=7, color='white',
                     fontweight='bold', clip_on=True)
        axes[0].text(v, bar.get_y() + bar.get_height() / 2, f' {v:.1f}',
                     va='center', ha='left', fontsize=7, color='black')
    nice_patch = mpatches.Patch(color='#2ecc71', label='NICE')
    nasty_patch = mpatches.Patch(color='#e74c3c', label='NASTY')
    axes[0].legend(handles=[nice_patch, nasty_patch], loc='lower right', fontsize=8)
    bars_win  = axes[1].barh(y_pos, wins, color='#3498db', label='Win')
    bars_draw = axes[1].barh(y_pos, draws, left=wins, color='#95a5a6', label='Seri')
    bars_lose = axes[1].barh(y_pos, loses, left=[w + d for w, d in zip(wins, draws)],
                              color='#c0392b', label='Lose')
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Average match per tournament')
    axes[1].set_title('Win / Draw / Loss')
    axes[1].legend(loc='lower right', fontsize=8)
    annot_left = axes[0].annotate("", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
                                   bbox=dict(boxstyle="round", fc="w", ec="black"),
                                   fontsize=8, visible=False)
    annot_right = axes[1].annotate("", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
                                    bbox=dict(boxstyle="round", fc="w", ec="black"),
                                    fontsize=8, visible=False)
    all_bar_groups = {
        axes[0]: (bars_left, annot_left),
        axes[1]: (list(bars_win) + list(bars_draw) + list(bars_lose), annot_right),
    }
    def on_move(event):
        redraw = False
        for ax, (bars, annot) in all_bar_groups.items():
            if event.inaxes != ax:
                if annot.get_visible():
                    annot.set_visible(False)
                    redraw = True
                continue
            found = False
            for idx in range(len(names)):
                bar = bars[idx % len(names)] if ax == axes[1] else bars[idx]
                contains, _ = bar.contains(event)
                if contains:
                    creator = strategies_creator.get(names[idx], 'Unknown')
                    annot.xy = (event.xdata, event.ydata)
                    annot.set_text(f"{names[idx]}\nCreator: {creator}")
                    annot.set_visible(True)
                    found = True
                    redraw = True
                    break
            if not found and annot.get_visible():
                annot.set_visible(False)
                redraw = True
        if redraw:
            fig.canvas.draw_idle()
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    plt.tight_layout()
    plt.savefig('tournament_result.png', dpi=150)
    plt.show()

def timeout_input_instant(second):
    start = time.time()
    while time.time() - start < second:
        if msvcrt.kbhit():
            char = msvcrt.getch().decode()
            return char
    return None

def many_name_strategies_function(strategy):
    strategy_code_name = strategy
    if strategy_code_name not in strategies:
        for i in range(len(strategies_many_name)):
            if strategy_code_name in strategies_many_name[i].split(' / '):
                return strategies_many_name[i]
    return strategy

def update_score(x, y):
    global game_score
    game_score[0] += x
    game_score[1] += y

def reset_internal_memory(self_index):
    self_payoff_matrix = game_payoff_matrix[self_index]
    self_avg_payoff_matrix = game_avg_payoff_matrix[self_index]
    return [
            0, # 0
            True, # 1
            ["Normal",0,0,None,False,0.25,np.zeros([4, 2]),0,3,False,0,[0,0],0,5,0,0], # 2
            [0,0], # 3
            0, # 4
            [0.3, 0, 0], # 5
            [0.5,0], # 6
            0, # 7
            [0,0], # 8
            [0,0.5], # 9
            0, # 10
            [0,0,0], # 11
            [0,0,0], # 12
            [0.5,0], # 13
            {"mode": "TFT", "matrix": np.zeros((2, 2))}, # 14
            [0, 0, 0, 0, 0], # 15
            [0,0,0,0], # 16
            0, # 17
            0, # 18
            0, # 19
            [[0,0,0,0,1,0],[0,0,0,0,0,0],0], # 20
            [[30,0],[30,0],[0,30],[0,30],0], # 21
            [0,0,0,0], # 22
            [[0,0,0,0],[0,0,0,0],[0,0,0,0],True,[0.0, 4.0, 6.0, 6.0, 8.0, 12.0],[[36.0, 0.0, 0.0, 0.0],[16.0, 12.0, 12.0, 0.0],[0.0, 18.0, 24.0, 0.0],[12.0, 12.0, 9.0, 9.0],[0.0, 16.0, 16.0, 12.0],[0.0, 0.0, 0.0, 48.0],]], # 23
            [0.1,0,0,0], # 24
            0, # 25
            0, # 26
            [0,0], # 27
            '2', # 28
            0, # 29
            0, # 30
            [
                {
                    "n": 155,
                    "min_n": 44,
                    "max_n": 302,
                    "adj": np.random.uniform(-0.15, 0.15, (155, 155)),
                    "volt": np.zeros(155, dtype=np.float64),
                    "th": np.random.uniform(0.9, 1.1, 155),
                    "ref": np.zeros(155, dtype=np.int32),
                    "a": 0.5,
                    "energy": 1000.0,
                    "lactate": 0.0,
                    "DA": 0.5, "CORT": 0.2, "SER": 0.5, "OXY": 0.5,
                    "ADR": 0.2, "NOR": 0.2, "TES": 0.3, "GAB": 0.5,
                    "END": 0.5, "ACH": 0.5, "VAS": 0.3, "ADE": 0.0,
                    "MEL": 0.0, "GLU": 0.7,
                    "day_tick": 0,
                    "is_sleeping": False,
                    "last_spike_time":np.full(155, -np.inf),
                    "decay factor": [0.75 ** i for i in range(10)],
                    "lead": 0
                }
            ], # 31
            0, # 32
            [0,0], # 33
            [
                {
                    ("CCCCC","CCCCC"):["CCCCC",100]
                },
                [0,("CCCCC","CCCCC"),0]
            ], # 34
            [False, 0, 0, [0] * 12, 0], # 35
            [[1,1,1,1,1],[0,0,0,0,0]], # 36
            0, # 37
            [
                {
                    ("CCCCC", "CCCCC"):1
                },
            [None,0]
            ], # 38
            [0,0,0.5,0,0,0], # 39
            [0,0.5,0], # 40
            [0,0,0.5], # 41
            0, # 42
            [
                [
                    {
                        ("C","C"):"C"
                    } for _ in range(64)
                ],
            [rnd.randint(0,8) for _ in range(64)],
            [rnd.randint(0,8) for _ in range(64)],
            [0 for _ in range(64)],
            [0 for _ in range(64)],
            [0 for _ in range(64)]
            ], # 43
            [
                {
                    ("C","C"):"C"
                },
                [
                    {
                        ("C","C"):"C"
                    },
                    {
                        ("C","C"):"C"
                    }
                ],
            [None,None,None]
            ], # 44
            [0,0,0,0], # 45
            [
                {
                    ("C","C"):[1,0]
                },
            [[None, 0.5],0.1]
            ], # 46
            0, # 47
            0, # 48
            0, # 49
            0, # 50
            [0,0], # 51
            0, # 52
            [0,0], # 53
            [0,0], # 54
            [0,0], # 55
            0, # 56
            [0,0,0,0,0], # 57
            [0,0,0,0,0], # 58
            [0,0,0,0,0], # 59
            [0,0,0,0,0], # 60
            [0,0,0,0,0], # 61
            [0,0,0,0,0], # 62
            0, # 63
            0, # 64
            [
                {
                    "n":24,
                    "input old val":[0, 0, 0, 0],
                    "input weight":[[rnd.uniform(-1,1) for _ in range(4)] for _ in range(24)],
                    "recurrent weight":[rnd.uniform(-1,1) for _ in range(24)],
                    "bias":[rnd.uniform(-1,1) for _ in range(24)],
                    "n val":[0 for _ in range(24)],
                    "n real val":[0 for _ in range(24)],
                    "n old val":[0 for _ in range(24)],
                    "n old real val":[0 for _ in range(24)],
                    "output weight":[[rnd.uniform(-1,1) for _ in range(24)], [rnd.uniform(-1,1) for _ in range(24)]],
                    "output bias":[rnd.uniform(-1,1), rnd.uniform(-1,1)],
                    "output val":[0.5, 0.5],
                    "output old val":[0.5, 0.5],
                    'scalar trace wi':[[0 for _ in range(4)] for _ in range(24)],
                    'scalar trace wr':[0 for _ in range(24)],
                    'scalar trace wo':[[0 for _ in range(24)], [0 for _ in range(24)]],
                    'old scalar trace wi':[[0 for _ in range(4)] for _ in range(24)],
                    'old scalar trace wr':[0 for _ in range(24)],
                    'old scalar trace wo':[[0 for _ in range(24)], [0 for _ in range(24)]],
                    'G':[0 for _ in range(24)]
                }
            ], # 65
            [0,1], # 66
            0, # 67
            0, # 68
            0, # 69
            [
                {
                    ("C", "C"):1
                },
                None
            ], # 70
            [2,0], # 71
            [0,0], # 72
            [0,0], # 73
            0, # 74
            0, # 75
            0, # 76
            [0,0], # 77
            [0,0.5], # 78
            1, # 79
            [0,0,0], # 80
            0, # 81
            [False,0,0,0,False,0], # 82
            0, # 83
            [2,[1] * 4,0,21,2,float("inf"),2,2], # 84
            [0,0], # 85
            0, # 86
            0, # 87
            0, # 88
            [0,0,5,0,1,1,{'C':0,'D':1},0], # 89
            0, # 90
            [{'CC':[0,0], 'CD':[0,0], 'DC':[0,0], 'DD':[0,0]}, 'CC', 0], # 91
            0, # 92
            [0 for _ in range(7)], # 93
            0, # 94
            0, # 95
            0, # 96
            0, # 97
            0, # 98
            0, # 99
            [
                {
                    "CCCCC":["TTT",100],
                    "CCCCD":["TTT",100],
                    "CCCDC":["TTT",100],
                    "CCDCC":["TTT",100],
                    "CDCCC":["TTT",100],
                    "DCCCC":["TTT",100],
                },
                [0,"CCCCC",0]
            ], # 100
            [0,0,0], # 101
            [0,0,0], # 102
            [0,0,0,0], # 103
            0.5, # 104
            0, # 105
            [0,0], # 106
            0, # 107
            0, # 108
            [0,0], # 109
            0, # 110
            0, # 111
            0, # 112
            [0,0], # 113
            [1,0,0], # 114
            0, # 115
            [0,0,0], # 116
            0.5, # 117
            [0,0,0], # 118
            [
                {
                    "C":[1,0],
                    "D":[1,1]
                },
                "C"
            ], # 119
            1, # 120
            [0,0,0,0,False,"Normal"], # 121
            0, # 122
            0, # 123
            [0,0], # 124
            [
                {
                    ("C", "C", "C"):0,
                    ("C", "C", "D"):0,
                    ("C", "D", "C"):0,
                    ("C", "D", "D"):0,
                    ("D", "C", "C"):0,
                    ("D", "C", "D"):0,
                    ("D", "D", "C"):0,
                    ("D", "D", "D"):0
                },
                [],
                0
            ], # 125
            0, # 126
            [0,0,0], # 127
            [1, False, 1, 0], # 128
            {
                "step": 1,
                "substp": 1,
                "bothd": 0,
                "titcnt": 0,
                "tatcnt": 0,
                "evil": 0,
                "n": 1,
                "f": 0,
                "ok": [0, 0, 0, 0],
                "totk": 0,
                "oldmov": 0,
                "count": 0,
            }, # 129
            0, # 130
            [False,[]], # 131
            [0,0], # 132
            0, # 133
            {"d_count": 0, "is_retaliating": False}, # 134
            {
                "s": 3,
                "w": 0,
                "q": 0.8,
            }, # 135
            [[], [0], []], # 136
            [
                [
                    {
                        # opp, self, my_sc > opp_sc, opp_sc > my_sc : prob to choose C
                        ("C", "C", "T", "F"):1,
                        ("D", "C", "T", "F"):1,
                        ("C", "D", "T", "F"):1,
                        ("D", "D", "T", "F"):1,
                        ("C", "C", "F", "F"):1,
                        ("D", "C", "F", "F"):0,
                        ("C", "D", "F", "F"):1,
                        ("D", "D", "F", "F"):0,
                        ("C", "C", "F", "T"):0,
                        ("D", "C", "F", "T"):0,
                        ("C", "D", "F", "T"):0,
                        ("D", "D", "F", "T"):0
                    }
                ],
                ["C"]
            ], # 137
            0, # 138
            [
                {
                    (("C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C"),("C","C", "C", "C", "C", "C", "C", "C", "C", "C", "C")):[0,0]
                },
            [[None, 0.5],0.1]
            ], # 139
            [
                {
                    (("C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C"),("C","C", "C", "C", "C", "C", "C", "C", "C", "C", "C")):[0,0]
                },
            [[None, 0.5],0.1]
            ], # 140
            [
                {
                    (("C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C"),("C","C", "C", "C", "C", "C", "C", "C", "C", "C", "C")):[0,0]
                },
            [[None, 0.5],0.1]
            ], # 141
            False, # 142
            ['0',1], # 143
            ["C",0,1,0], # 144
            [3,-10], # 145
            0, # 146
            [0,0], # 147
            0, # 148
            [0,0,0,0], # 149
            0, # 150
            0, # 151
            0, # 152
            {
                "z": 0,
                "q6": 0.5,
                "s": 0,
                "h": 0,
            }, # 153
            [7,0], # 154
            ["Normal",0], # 155
            [{'C': 0, 'D': 1}, {'C': 0, 'D': 1}, False], # 156
            True, # 157
            {
                "x": 0.999,
                "px": 0.001,
                "y": 0.001,
                "py": 0.999,
                "z": 0.999,
                "pz": 0.001,
                "w": 0.001,
                "pw": 0.999,
                "qc": [1.999, 1.999, 0.001, 0.001],
                "qn": [2.0, 2.0, 2.0, 2.0],
                "ipol": [
                    [0, 0, 0, 0],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [0, 1, 1, 1],
                    [0, 1, 1, 1],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 1, 0, 0],
                    [1, 0, 1, 0],
                ],
                "iold": 0,
                "n": 0,
            }, # 158
            {
                "jz": 0,
            }, # 159
            {
                "j0": 0.0,
                "f0": 0.0,
                "j2": 0,
                "j4": 0,
                "j8": 0.0,
                "f4": 0,
                "f8": 0.0,
                "f1": 0,
                "c": 0,
                "d": 0,
                "t": 0,
                "i1": 0,
                "i2": 0,
                "i3": 0,
                "i4": 0,
            }, # 160
            [0,0], # 161
            {
                "mode": 0,
                "inod": 0,
            }, # 162
            {
                "alpha": 1.0,
                "beta": 0.0,
                "iold": 0,
                "qca": 0,
                "qna": 0,
                "qcb": 0,
                "qnb": 0,
                "mutdef": 0,
            }, # 163
            {
                "lastd": 0,
                "diff": 0,
                "totd": 0,
            }, # 164
            {
                "d8": 0,
                "d9": 0,
            }, # 165
            {
                "x": 0.75,
                "i5": 0,
                "d4": 0.0,
                "i1": 0,
                "i2": 0,
                "i3": 1,
            }, # 166
            {
                "jold": 0,
                "iran": int(23 * rnd.random() + 1),
            }, # 167
            {
                "good": 1.0,
                "bad": 0.0,
                "past": 0,
                "totcop": 0,
                "totdef": 0,
                "nice1": 0,
                "nice2": 0,
            }, # 168
            {
                "sc": [0] * 6,
                "sl": [1] * 6,
                "st": [0] * 5,
                "gt": [0] * 5,
                "tm": [0] * 6,
                "cn": 10,
                "csrc": 5,
                "mylm": 1,
                "hlm": 0,
            }, # 169
            {
                "s": 1,
                "f": 0,
                "d": 0,
                "c": 0,
            }, # 170
            {
                "opdef": 0,
                "stdef": 0,
                "dl": 0.20,
                "coops": 0,
                "okdef": True,
                "mydef": False,
                "nodef": 0,
                "nd": 12,
            }, # 171
            {
                "ia": 0,
                "ib": 0,
            }, # 172
            False, # 173
            0, # 174
            [{}], # 175
            False, # 176
            [
                0,
                {
                    'wondering': [
                        'Why is [REPLACE] on your mind?',
                        'Hmm, [REPLACE] keeps crossing your mind, doesn\'t it?',
                        'Since when have you been thinking about [REPLACE]?'
                    ],
                    'kill': [
                        'PLEASE DON\'T KILL [REPLACE]! :-(',
                        'KILLING [REPLACE] IS BAD! :-(',
                    ],
                    'hate': [
                        'Why do you hate [REPLACE]?',
                        'What makes you hate [REPLACE] so much? >:-(',
                        'Hate is a heavy thing, why [REPLACE]?'
                    ],
                    'love': [
                        'So, you love [REPLACE]?',
                        'Wow, in love with [REPLACE]? Tell me all about it! :-)',
                        'Love is beautiful, what does it feel like to love [REPLACE]?'
                    ],
                    'like': [
                        'Okay, so you like [REPLACE]?',
                        'Cool! Liking [REPLACE] is fun!',
                        'Why do you like [REPLACE]?'
                    ],
                    'miss': [
                        'You miss [REPLACE], don\'t you? :-(',
                        'What is it like missing [REPLACE]?',
                        'How long have you been missing [REPLACE]?'
                    ],
                    'fear': [
                        'What makes you afraid of [REPLACE]?',
                        'It\'s perfectly normal to be afraid of [REPLACE]—want to tell me about it?',
                        'If [REPLACE] weren\'t scary, what would you do?'
                    ],
                    'sad': [
                        'Why are you sad about [REPLACE]? :-(',
                        'I\'m sorry to hear about [REPLACE]',
                        'When you\'re sad about [REPLACE], what usually makes you feel better?'
                    ],
                    'happy': [
                        'Wow, you\'re happy because of [REPLACE], right? I\'m happy for you! :-)',
                        'What makes you happiest about [REPLACE]?',
                        'Glad to hear you\'re happy about [REPLACE]!'
                    ],
                    'angry': [
                        'Are you angry because of [REPLACE]? >:-(',
                        'Being angry about [REPLACE] is exhausting, you know—how do you want to let it out?',
                        'Okay, take a deep breath first. Tell me slowly about being angry over [REPLACE].'
                    ],
                    'tired': [
                        'Tired because of [REPLACE], huh? You should take a break :-(',
                        'If you\'re tired because of [REPLACE], what makes you want to give up?',
                        'Being tired is a sign that you\'ve been fighting hard regarding [REPLACE]'
                    ],
                    'Alone': [
                        'Why do you feel alone about [REPLACE]?',
                        'Sometimes it\'s not good to be alone, especially about [REPLACE] :-(',
                        'When you feel like you\'re [REPLACE], what do you usually do?'
                    ],
                    'we are': [
                        'Yes, we are [REPLACE]',
                        'We are [REPLACE]? Interesting, tell me more.'
                    ],
                    'dreaming': [
                        'Dreaming about [REPLACE]? A nightmare or a pleasant dream?',
                        'What if the dream about [REPLACE] came true?',
                        'Interesting, tell me more about that dream involving [REPLACE].'
                    ],
                    'friend': [
                        'What about the friend who [REPLACE]?',
                        'Is that friend who [REPLACE] a close friend?',
                        'What makes you think about the friend who [REPLACE]?'
                    ],
                    'because': [
                        'Okay, so you think that\'s because [REPLACE]?',
                        'So the main reason is because [REPLACE], right?',
                        'Hmm, because [REPLACE]... then what?'
                    ],
                    'yes': ['Alright', 'Okay, go on?', 'Got it, what next?', 'Mhm, I\'m listening'],
                    'hello': [f'Hello there!', f'Hi there!', f'Yo, friend! What\'s up?'],
                }
            ], # 177
            [100,1,0,False,None,0,{("C", "C"):[0,0], ("C", "D"):[0,0], ("D", "C"):[0,0], ("D", "D"):[0,0]}], # 178
            [0.5,0.5], # 179
            [
                [
                    {
                        (("C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C"),("C","C", "C", "C", "C", "C", "C", "C", "C", "C", "C")):[0,0]
                    },
                [[None, 0.5],0.1]
                ],
                [
                    {
                        (("C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C"),("C","C", "C", "C", "C", "C", "C", "C", "C", "C", "C")):[0,0]
                    },
                [[None, 0.5],0.1]
                ]
            ], # 180
            [100,1,False,0], # 181
            [100,1,False,None,0,{("C", "C"):[0,0], ("C", "D"):[0,0], ("D", "C"):[0,0], ("D", "D"):[0,0]}], # 182
            [100,1,False,None,0,{("C", "C"):[0,0], ("C", "D"):[0,0], ("D", "C"):[0,0], ("D", "D"):[0,0]}], # 183
            [100,1,False,None,0,{("C", "C"):[0,0], ("C", "D"):[0,0], ("D", "C"):[0,0], ("D", "D"):[0,0]},0], # 184
            [100,1,False,0,0], # 185
            [100,1,False,None,0,{("C", "C"):[0,0], ("C", "D"):[0,0], ("D", "C"):[0,0], ("D", "D"):[0,0]}], # 186
            0, # 187
            {("C", "C"):[0,0], ("C", "D"):[0,0], ("D", "C"):[0,0], ("D", "D"):[0,0]}, # 188
            0, # 189
            100, # 190
            0, # 191
            0, # 192
            0, # 193
            0, # 194
            {
                "ncc": 0,
                "ncd": 0,
                "ndc": 0,
                "ndd": 0,
                "kount": 0,
                "mytwin": 0,
                "iold1": 0,
                "iold2": 0,
            }, # 195
            [0,False,0], # 196
            {
                "l4": [[0.0, 0.0] for _ in range(8)],  # Matriks frekuensi 8x2
                "t0": 0,
                "t4": 0,
                "t5": 0,
                "t6": 25,
                "t8": 0,
                "t9": 5,
                "s1": 0,
            }, # 197
            0, # 198
            [0,0], # 199
            0, # 200
            0, # 201
            0, # 202
            {('C', 'C'): 0, ('C', 'D'): 0, ('D', 'C'): 0, ('D', 'D'): 0}, # 203
            [False,0,20,{('C', 'C'): 0, ('C', 'D'): 0, ('D', 'C'): 0, ('D', 'D'): 0}], # 204
            0, # 205
            0, # 206
            [], # 207
            '1', # 208
            '1', # 209
            0, # 210
            [1, False, 1, 0], # 211
            [1.0,0.0,0,0,0,0], # 212
            0, # 213
            0, # 214
            0, # 215
            0, # 216
            {
                "ema_trust": 1.0,
                "d_streak": 0,
                "history": [],
                "regret_count": 0,
                "pattern_score": 0,
                "pattern_window": 0,
                "punish_cooldown": 0,
                "last_my_action": 'C'
            }, # 217
            {
                "n":50,
                "old input": [0, 0],
                "input weight":[[rnd.uniform(-1,1) for _ in range(2)] for _ in range(50)],
                "bias":[rnd.uniform(-1,1) for _ in range(50)],
                "n val":[0.5 for _ in range(50)],
                "output weight":[[rnd.uniform(-1,1) for _ in range(50)], [rnd.uniform(-1,1) for _ in range(50)]],
                "output bias":[rnd.uniform(-1,1), rnd.uniform(-1,1)],
                "output val":[0.5, 0.5],
            }, # 218
            {
                "n":2,
                "time":0,
                "input weight":[[rnd.uniform(-1,1) for _ in range(2)] for _ in range(2)],
                "bias":[rnd.uniform(-1,1) for _ in range(2)],
                "n val":[0.5 for _ in range(2)],
                "output weight":[[rnd.uniform(-1,1) for _ in range(2)], [rnd.uniform(-1,1) for _ in range(2)]],
                "output bias":[rnd.uniform(-1,1), rnd.uniform(-1,1)],
                "output val":[0.5, 0.5],
                "best input weight":[[0 for _ in range(2)] for _ in range(2)],
                "best bias":[0 for _ in range(2)],
                "best output weight":[[0 for _ in range(2)], [0 for _ in range(2)]],
                "best output bias":[0, 0],
                "score":0,
                "high score":1,
                "reward":1,
            }, # 219
            {
                "pattern_score": 0,
                "trust": 1.0,
                "history": [],
                "d_streak": 0,
                "mode": "observing",
                "forgiveness_chance": 0.3
            }, # 220
            [0,0,0], # 221
            [0,0,0], # 222
            0, # 223
            {
                "trust": 1.0,
                "deal_broken": False,
                "coop_count": 0,
                "defect_count": 0
            }, # 224
            {
                "my_bluff_count": 0,
                "opp_bluff_count": 0,
                "opp_coop_words": 0,
                "opp_betrayals": 0
            }, # 225
            [0,[0 for _ in range(5)]], # 226
            {
                "acumulation":{'CC':0,'CD':0,'DC':0,'DD':0},
                "tree":{'CC':0.5,'CD':0.5,'DC':0.5,'DD':0.5},
                "total":{'CC':0,'CD':0,'DC':0,'DD':0},
            }, # 227
            [{'CC':[0,0], 'CD':[0,0], 'DC':[0,0], 'DD':[0,0]}, 'CC'], # 228
            0, # 229
            {
                'RRR':(self_payoff_matrix['CC'] - self_payoff_matrix['CD']) / (self_payoff_matrix['DC'] - self_payoff_matrix['CC']), # (R - S) / (T - R)
                'Tolerant':50 / (self_payoff_matrix['CC'] - self_payoff_matrix['CD']), # total tolerant loss score / (R - S)
                'Win round total':0,
                'Lose round total':0,
                'Highest score ever get':self_avg_payoff_matrix, # (T + R + P + S) / 4
                'Lowest score ever get':self_avg_payoff_matrix, # (T + R + P + S) / 4
                'G':[0,0],
                'B':[0,0],
                'Total poin opp C':0,
                'Total poin opp D':0,
                'Old score':0,
                'ALLD':False
            }, # 230
            [{'C':0,'D':0}, 0], # 231
            0, # 232
            0, # 233
            0, # 234
            None, # 235
            None, # 236
            0, # 237
            [0,False,0,0,0], # 238
            0, # 239
            [0,0], # 240
            [0.5,0.5,abs(max(self_payoff_matrix['DC'],self_payoff_matrix['CC'],self_payoff_matrix['DD'],self_payoff_matrix['CD']) / 3),0.0,0.5,0], # 241
            [[0]*20,False], # 242
            [1,0,['C','C','D']], # 243
            {
                "best pattern 1":[['C','C','D'], 0],
                "best pattern 2":[['C','C','D'], 0],
                "now pattern":['C','C','D'],
                "pattern score":0,
                "sum old index":0,
                "old score":0
            }, # 244
            {
                'Rd': {('C', 'C'): 1, ('C', 'D'): 1, ('D', 'C'): 0, ('D', 'D'): 0},
                'Rc': {},
                'Pi': {('C', 'C'): 1, ('C', 'D'): 1, ('D', 'C'): 0, ('D', 'D'): 0},
                'violation_counts': {},
                'reject_threshold': 3,
                'violation_threshold': 4,
                'promotion_threshold': 3,
                'tree_depth': 5,
                'v': 0,
                'alpha': 0.75,
                'history_by_cond': {
                    ('C', 'C'): ([1], [1]),
                    ('C', 'D'): ([1], [1]),
                    ('D', 'C'): ([0], [1]),
                    ('D', 'D'): ([0], [1]),
                }
            }, # 245
            [0,0], # 246
            '0', # 247
            '1', # 248
            '0', # 249
            '1', # 250
            '1', # 251
            '1', # 252
            '1', # 253
            '1', # 254
            '1', # 255
            '0', # 256
            '0', # 257
            '0', # 258
            '0', # 259
            '0', # 260
            '0', # 261
            '2', # 262
            0, # 263
            0, # 264
            [0.25,{},"",""], # 265
            '', # 266
            '', # 267
            '', # 268
            ["C",max(round(rnd.randint(1,50) * rnd.random()),1),0], # 269
            False, # 270
            [False, 0], # 271
            False, # 272
            False, # 273
            [False, 0], # 274
            False, # 275
            0, # 276
            ["Nice",10,-10,0,0], # 277
            [0, [[1.0]], [[1.0]], [0.5]], # 278
            3, # 279
            [1, 1, 0, 0], # 280
            [0,0,0], # 281
            False, # 282
            [0, 0], # 283
            [0, 0, 0, 0], # 284
            None, # 285
            [0, 0, 0, 0, (1 + mth.sqrt(5)) / 2], # 286
            [0, 0, 0, 0, mth.pi], # 287
            [0, 0, 0, 0, mth.e], # 288
            [max(min(rnd.random() * ((rnd.random() * 0.5) + 1), 1), 0), max(min(rnd.random() * (1 - (rnd.random() * 0.5)), 1), 0)], # 289
            ["TFT", 3, 0], # 290
            [], # 291
            [[], [0], []], # 292
            {206:0,157:True,284:[0, 0, 0, 0],283:[0, 0],282:False}, # 293
            1.0, # 294
            [False, 0], # 295
            0, # 296
            0, # 297
            [
                [
                    'C',
                    'C',
                    'D',
                    'C',
                    'D',
                    'D',
                    'D',
                    'C',
                    'C',
                    'D',
                    'C',
                    'D',
                    'C',
                    'C',
                    'D',
                    'C',
                    'D',
                    'D',
                    'C',
                    'D',
                ],
                0,
                0,
                False
            ], # 298
            0, # 299
            [
                {
                    (("C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C"),("C","C", "C", "C", "C", "C", "C", "C", "C", "C", "C")):[0,0]
                },
            [[None, 0.5],0.1]
            ], # 300
            {('C', 'C'): 0, ('C', 'D'): 0, ('D', 'C'): 0, ('D', 'D'): 0}, # 301
            {('C', 'C'): 0, ('C', 'D'): 0, ('D', 'C'): 0, ('D', 'D'): 0}, # 302
            [False,0,15,{('C', 'C'): 0, ('C', 'D'): 0, ('D', 'C'): 0, ('D', 'D'): 0}], # 303
            [False,0,20,{('C', 'C'): 0, ('C', 'D'): 0, ('D', 'C'): 0, ('D', 'D'): 0}], # 304
            ['0',1], # 305
            0, # 306
            [3, 8, 0, 0], # 307
            False, # 308
            0, # 309
            [round(max(rnd.randint(1, 10) * rnd.random(), 1)), round(max(rnd.randint(1, 10) * rnd.random(), 1)), 0], # 310
            False, # 311
            0, # 312
            [0, 0], # 313
            (lambda n=4: {"num_states": n, "transitions": {s: {"C": [x/sum(wc) for wc in [[rnd.random() for _ in range(n)]] for x in wc], "D": [x/sum(wd) for wd in [[rnd.random() for _ in range(n)]] for x in wd]} for s in range(n)}, "state_outputs": [rnd.choice(["C", "D"]) for _ in range(n)], "current_state": 0})(), # 314
            (lambda n=4: {"num_states": n, "transitions": {s: {"C": [x/sum(wc) for wc in [[rnd.random() for _ in range(n)]] for x in wc], "D": [x/sum(wd) for wd in [[rnd.random() for _ in range(n)]] for x in wd]} for s in range(n)}, "state_outputs": [rnd.choice(["C", "D"]) for _ in range(n)], "current_state": 0})(), # 315
            (lambda n=4: {"num_states": n, "transitions": {s: {"C": [x/sum(wc) for wc in [[rnd.random() for _ in range(n)]] for x in wc], "D": [x/sum(wd) for wd in [[rnd.random() for _ in range(n)]] for x in wd]} for s in range(n)}, "state_outputs": [rnd.choice(["C", "D"]) for _ in range(n)], "current_state": 0})(), # 316
            [0, 0], # 317
            0, # 318
            '1', # 319
            '1', # 320
            '1', # 321
            0, # 322
            [0, 0, 0, 0], # 323
            False, # 324
            [False, []], # 325
            1.0, # 326
            0, # 327
            [{'C': 0, 'D': 0}, {'C': 0.0, 'D': 0.0}], # 328
            0, # 329
            0, # 330
            0, # 331
            [22, 35, 0, 0], # 332
            [[], [], None, [], None], # 333
            [[], [], False], # 334
            [0, 0], # 335
            [True, 1.0, 1.0, 1.0, 1.0], # 336
            [{'CC':1,'CD':0,'DC':1,'DD':0},{'CC':1,'CD':0,'DC':1,'DD':0}], # 337
            [1, 0, 0, 0], # 338
            [
                {
                    'C': {'C': 0.0, 'D': 0.0},
                    'D': {'C': 0.0, 'D': 0.0}
                },
                0
            ], # 339
            [
                {
                    (("C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C"),("C","C", "C", "C", "C", "C", "C", "C", "C", "C", "C")):[0,0]
                },
            [[None, 0.5],0.1]
            ], # 340
            [
                {
                    (("C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C"),("C","C", "C", "C", "C", "C", "C", "C", "C", "C", "C")):[0,0]
                },
            [[None, 0.5],0.1]
            ], # 341
            [0.5, 0.5, 0.5], # 342
            [
                {
                    (("C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C"),("C","C", "C", "C", "C", "C", "C", "C", "C", "C", "C")):[0,0]
                },
            [[None, 0.5],0.1]
            ], # 343
            [0.5,0.5], # 344
            [{'CC':[0,0], 'CD':[0,0], 'DC':[0,0], 'DD':[0,0]}, 'CC', 0], # 345
            {'CC':[0,0], 'CD':[0,0], 'DC':[0,0], 'DD':[0,0]}, # 346
            False, # 347
            False, # 348
            False, # 349
            False, # 350
            False, # 351
            False, # 352
            False, # 353
            False, # 354
            False, # 355
            False, # 356
            False, # 357
            0, # 358
            0, # 359
            '', # 360
            0, # 361
            0, # 362
            0, # 363
            0, # 364
            0, # 365
            ['CCCC','CCCD',0,0], # 366
            ['CCCD','CCCC',0,0], # 367
            ['Step 1', 0], # 368
            [
                [1, 11, 21, 31, 3, 2, 13, 23, 22, 33, 5, 4, 15, 25, 7, 6, 17, 27, 26, 37],
                [1, 0, 11, 10, 21, 20, 31, 30, 3, 2, 13, 12, 23, 22, 33, 32, 5, 4, 15, 14, 25, 24, 35, 34, 7, 6, 17, 16, 27, 26, 37, 36],
                1,
                1,
                0,
                15,
                0
            ], # 369
            -1, # 370
            ["Phase 1", [0, 0, 0, 0]], # 371
            set(), # 372
            False, # 373
            False, # 374
            0, # 375
            '', # 376
            min(rnd.random() * (rnd.random() + 1), 1), # 377
            1, # 378
            [min(rnd.random() * (rnd.random() + 1), 1), min(rnd.random() * (rnd.random() + 1), 1), rnd.random() * rnd.random(), rnd.random() * rnd.random()], # 379
            {"jold": 0, "a": 0, "b": 0, "c": 0, "d": 0, "e": 0}, # 380
            {
                "iarray": [0, 0, 0, 0, 0],
                "ipo2": [2, 4, 3, 5, 1],
                "ipo1": 1,
                "ichan": 1,
                "kold": 0,
                "k5": 0,
                "klast": 0,
            }, # 381
            [0, 0, 0, 0], # 382
            [0, 0, 0, 0], # 383
            0, # 384
            [rnd.randint(1, 25), rnd.randint(1, 25), rnd.randint(1, 25), rnd.randint(1, 25)], # 385
            [0, {'CC':[0,0], 'CD':[0,0], 'DC':[0,0], 'DD':[0,0]}], # 386
            {
                "n": 0
            }, # 387
            {
                "iaggd": 4,
                "idunu": 0,
                "idunb": 0,
                "ipayb": 8,
                "itest": 1,
                "ipost": 0,
            }, # 388
            {
                "alpha": 1.0,
                "beta": 0.3,
                "iold": 0,
                "qca": 0.0,
                "qna": 0.0,
                "qcb": 0.0,
                "qnb": 0.0,
                "k74r_val": 0,
                "jsw": 0,
                "js4": 0,
                "js11": 0,
                "jr": 0,
                "jl": 0,
                "jt": 0,
                "jsm": 1,
            }, # 389
            {
                "alpha": 1.0,
                "beta": 0.3,
                "iold": 0,
                "qca": 0.0,
                "qna": 0.0,
                "qcb": 0.0,
                "qnb": 0.0,
                "k74dummy": 0,
                "jsw": 0,
                "js4": 0,
                "js11": 0,
                "jr": 0,
                "jl": 0,
                "jt": 0,
                "jsm": 1,
            }, # 390
            {
                "jstr": 3,
                "ktry": 0,
                "kexp": [100, 100, 100, 100, 100],
                "ki": 0,
            }, # 391
            '1', # 392
            '1', # 393
            '1', # 394
            '1', # 395
            '1', # 396
            '1', # 397
            '1', # 398
            '1', # 399
            '1', # 400
            '1', # 401
            '1', # 402
            '1', # 403
            {}, # 404
            [0,0], # 405
            [0, 0, 0, 0], # 406
            {
                "history of move": [0, 0, 0, 0],
                "last move": 0,
                "game status": 0,
                "random counter": 0
            }, # 407
            False, # 408
            [False, False], # 409
            {}, # 410
            [False, False], # 411
            0.5, # 412
            [{'C':0.0, 'D':0.0}, -1.0], # 413
            [{'C':0.0, 'D':0.0}, -1.0], # 414
            {'s':0.5, 'l':3.0, 'SCO':{'C':0.0, 'D':0.0}}, # 415
            {
                "state": "COOPERATIVE",
                "is_red_line": False,
                "epoch_step": 0,
                "debt": 0,
                "systemic": 0,
                "queue": [],
                "opp_len": 0,
                "opp_defects": 0,
                "my_D": 0,
                "my_C": 0,
                "opp_coops_after_my_D": 0,
                "opp_defects_after_my_C": 0,
                "opp_coops_after_my_C": 0,
                "opp_defects_after_my_D": 0,
                "last_my": 'C',
                "late_defects": 0,
                "opp_last3": [],
                "echo_forgive": 0,
                "one_shot_forgives": 0,
                "clean_peace": 0,
                "deadlock": 0,
                "probe_fired": False,
                "warmth": rnd.uniform(0.40, 0.60),
                "target_warmth": 0.50,
                "intent": "normal",
                "tactical": "adaptive",
                "contrite": False,
                "last_D_reason": None,
                "consecutive_forgiven": 0,
                "apology_mode": False,
                "apology_steps": 0,
                "apology_tries": 0,
                "apology_max_tries": 2,
                "first_D_was_noise": False,
                "apology_opp_moves": [],
                "forgiveness_exploiter": 0,
                "opening_d": False,
                "cc_pairs": 0,
                "unprovoked_d": 0,
                "p_noise_est": 0.0,
                "intended_prev": 'C',
                "bad_standing": False,
                "bad_standing_turns": 0,
                "red_line_cooldown": 0,
                "nl_evidence": 0,
                "nl_window": [],
                "nl_scheduled": [],
                "nl_debt_ledger": 0,
                "nl_pardoned": 0,
                "nl_balance": 0,
                "nl_harvest_left": 0,
                "nl_harvest_size": 0,
                "nl_harvest_postponed": 0,
                "nl_harvest_scheduled_at": 0,
                "nl_my_flips": 0,
                "nl_my_obs": 0,
                "nl_provoked_until": 0,
                "nl_pardon_step": 0,
                "pending_reason": None,
                "step_hint": 0,
                "harvest_window": rnd.randint(3, 6),
            }, # 416
            {
                "state": "COOPERATIVE",
                "is_red_line": False,
                "epoch_step": 0,
                "debt": 0,
                "systemic": 0,
                "queue": [],
                "opp_len": 0,
                "opp_defects": 0,
                "opp_coops_after_my_D": 0,
                "my_D": 0,
                "last_my": 'C',
                "late_defects": 0,
            }, # 417
            {'Alpha':1.0, 'Beta':1.0, 'Forgiveness Threshold':0.45, 'Uncertainty Factor':2.5}, # 418
            {'noise':0.0, 'n_tft_would_c':0, 'n_d_when_tft_would_c':0, 'z':0.0}, # 419
            {
                "discount_factor": 0.99,
                "ewma_CC": [1.0, 1.0],
                "ewma_CD": [1.0, 1.0],
                "ewma_DC": [0.0, 1.0],
                "ewma_DD": [0.0, 1.0],
                "opp_model": np.array([1.0, 0.0, 1.0, 0.0]),
                "my_policy": np.array([1.0, 0.0, 1.0, 0.0]),
            }, # 420
            {
                "iso_mem": {
                    "discount_factor": 0.99,
                    "ewma_CC": [1.0, 1.0],
                    "ewma_CD": [1.0, 1.0],
                    "ewma_DC": [0.0, 1.0],
                    "ewma_DD": [0.0, 1.0],
                    "opp_model": np.array([1.0, 0.0, 1.0, 0.0]),
                    "my_policy": np.array([1.0, 0.0, 1.0, 0.0]),
                },
                "n_tft_would_c": 0,
                "n_d_when_tft_would_c": 0,
                "z": 0.0,
                "playing_iso": False,
                "reward_history": [],
            }, # 421
            0.5, # 422
            [[0, 0], 'COSMOS', {139:[{(("C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C"),("C","C", "C", "C", "C", "C", "C", "C", "C", "C", "C")):[0,0]},[[None, 0.5],0.1]]}], # 423
            [0], # 424
            [0], # 425
            [0], # 426
            {206:0,157:True,284:[0, 0, 0, 0],283:[0, 0],282:False,205:0}, # 427
            {292:[[], [0], []]}, # 428
            {136:[[], [0], []]}, # 429
            [0, 0, 0, 0, constant('Champernowne')], # 430
            [0, 0, 0, 0, constant('Tau')], # 431
            [0, 0, 0, 0, constant('Apery')], # 432
            [0, 0, 0, 0, constant('Euler-Mascheroni')], # 433
            [0, 0, 0, 0, constant('Pythagoras')], # 434
            'C', # 435
            'C', # 436
            'C', # 437
            'C', # 438
            'C', # 439
            'C', # 440
            'Visitor', # 441
            'Visitor', # 442
            {"W":1,"B":0,"CORT":0,"DOPA":0,"F":0}, # 443
           ]

def reset_immortal_memory(self_index):
    self_payoff_matrix = game_payoff_matrix[self_index]
    self_avg_payoff_matrix = game_avg_payoff_matrix[self_index]
    return [
            {'Trust':rnd.random(), 'Trustworthiness':rnd.random()}, # 0
            [], # 1
            [], # 2
            {}, # 3
            {}, # 4
            {'Trust':rnd.random(), 'Trustworthiness':rnd.random()}, # 5
            {'Trust':rnd.random(), 'Trustworthiness':rnd.random()}, # 6
            {'Trust':rnd.random(), 'Trustworthiness':rnd.random(), 'Memory':{}}, # 7
            {'Trust':rnd.random(), 'Trustworthiness':rnd.random(), 'Memory':{}}, # 8
           ]

def cycle_detec(history, opp_index, min_ = 1, max_ = 12, offset = 0):
    inp = [history[-i][opp_index] for i in range(offset, len(history) + 1)] 
    new_max_size = min(len(inp) // 2, max_)
    for i in range(min_, new_max_size + 1):
        has_cycle = True
        cycle = tuple(inp[:i])
        for j, elem in enumerate(inp):
            if elem != cycle[j % len(cycle)]:
                has_cycle = False
                break
        if has_cycle:
            return cycle
    return None

def flip(the_choice):
    return 'C' if the_choice == 'D' else 'D'

def rnd_prob(prob):
    return 'C' if rnd.random() <= prob else 'D'

def frequency(myAction, opResponse, sample, opp_index):
    self_index = 1 - opp_index
    unziped_memory = list(zip(*sample))
    self_history, opp_history = unziped_memory[self_index], unziped_memory[opp_index]
    i = len(sample)-1
    N = 0;  M = 0
    while i > 0:
        if self_history[-(i+1)] == myAction:
            N += 1
            if opp_history[-i] == opResponse:
                M += 1
        i -= 1
    if N == 0:  return None
    else:  return float(M) / float(N)

def choice_majority(samples, tie = 'C'):
    A = samples.count('C')
    if A != (len(samples) / 2):
        return 'C' if A > (len(samples) / 2) else 'D'
    else:
        return tie

def transformator(action, category, opp_index, history):
    opp_history = [m[opp_index] for m in history]
    if category == 'NICE':
        if 'D' not in opp_history:
            return 'C'
    if category == 'NASTY':
        if 'D' not in opp_history:
            return 'D'
    return action

def clone_opp(if_error, strategy, history, opp_index, opp_name, internal_memory, immortal_memory, chats, next_chat, score, population):
    self_index = 1 - opp_index
    if opp_name not in strategies_inf_loop_maker:
        C = call_strategy(opp_name, history, self_index, strategy, internal_memory, immortal_memory, chats, next_chat, score, population)
        if C in ('C', 'D', '[OUT]'):
            return C
    return if_error

game_payoff_matrix = [{'CC':3, 'DD':1, 'CD':0, 'DC':5}, {'CC':3, 'DD':1, 'CD':0, 'DC':5}]
game_avg_payoff_matrix = [(game_payoff_matrix[0]['CC'] + game_payoff_matrix[0]['DD'] + game_payoff_matrix[0]['CD'] + game_payoff_matrix[0]['DC']) / 4, (game_payoff_matrix[1]['CC'] + game_payoff_matrix[1]['DD'] + game_payoff_matrix[1]['CD'] + game_payoff_matrix[1]['DC']) / 4]
strat_memory = [reset_internal_memory(0), reset_internal_memory(1)]
strat_immortal_memory = [reset_immortal_memory(0), reset_immortal_memory(1)]

def call_strategy(strategy, history, opp_index, opp_name, internal_memory, immortal_memory, chats, next_chat, score, population):
    self_index = 1 - opp_index
    internal_memory = internal_memory
    coop_prob = (strategies_coop_prob.get(strategy, 0.5), strategies_coop_prob.get(opp_name, 0.5))
    categories = (strategies_category.get(strategy, 'NETRAL'), strategies_category.get(opp_name, 'NETRAL'))
    self_payoff_matrix = game_payoff_matrix[self_index]
    self_avg_payoff_matrix = game_avg_payoff_matrix[self_index]
    opp_payoff_matrix = game_payoff_matrix[opp_index]
    opp_avg_payoff_matrix = game_avg_payoff_matrix[opp_index]
    # Strategies That Not From This Tournament
    if strategy == 'First By Tit For Tat':
        if not len(history):
            return 'C'
        return history[-1][opp_index]
    elif strategy == 'First By Joss / Naive Prober':
        if not len(history):
            return 'C'
        if rnd.random() <= 0.1:
            return 'D'
        else:
            return history[-1][opp_index]
    elif strategy == 'First By Grudger / Grim Trigger':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'D' or internal_memory[0] == 1:
            internal_memory[0] = 1
            return 'D'
        else:
            return 'C'
    elif strategy == 'Second By Gladstein / Tester':
        if not len(history):
            return 'D'
        if internal_memory[1]:
            if history[-1][opp_index] == 'D':
                internal_memory[1] = False
                return 'C'
            cooperation_ratio = sum((history[i][self_index] == 'C') for i in range(len(history))) / len(history)
            if cooperation_ratio > 0.5:
                return 'D'
            return 'C'
        else:
            return history[-1][opp_index]
    elif strategy == 'Sample / Second By Tit For 2 Tats':
        if (len(history)) < 2:
            return 'C'
        if history[-1][opp_index] == history[-1-1][opp_index] and history[-1][opp_index] == 'D':
            return 'D'
        else:
            return 'C'
    elif strategy == 'Second By Harrington':
        if len(history) > 0 and history[-1][opp_index] == 'C':
            internal_memory[2][15] += 1
        def try_return(to_return, lower_flags=True, inc_parity=False):
            """
            This will return to_return, with some end-of-turn logic.
            """
            if lower_flags and to_return == 'C':
                internal_memory[2][7] -= 1
                internal_memory[2][8] += 1
            if inc_parity and to_return == 'D':
                internal_memory[2][11][internal_memory[2][12]] += 1
            return to_return
        def calculate_chi_squared(turn):
            """
            Pearson's Chi Squared statistic = sum[ (E_i-O_i)^2 / E_i ], where O_i
            are the observed matrix values, and E_i is calculated as number (of
            defects) in the row times the number in the column over (total number
            in the matrix minus 1).  Equivalently, we expect we expect (for an
            independent distribution) the total number of recorded turns times the
            portion in that row times the portion in that column.
            In this function, the statistic is non-standard in that it excludes
            summands where E_i <= 1.
            """
            denom = turn - 2
            expected_matrix = (
                np.outer(
                    internal_memory[2][6].sum(axis=1), internal_memory[2][6].sum(axis=0)
                )
                / denom
            )
            chi_squared = 0.0
            for i in range(4):
                for j in range(2):
                    expect = expected_matrix[i, j]
                    if expect > 1.0:
                        chi_squared += (
                            expect - internal_memory[2][6][i, j]
                        ) ** 2 / expect
            return chi_squared
        def detect_random(turn):
            """
            We check if the top-left cell of the matrix (corresponding to all
            Cooperations) has over 80% of the turns.  In which case, we label
            non-random.
            Then we check if over 75% or under 25% of the opponent's turns are
            Defections.  If so, then we label as non-random.
            Otherwise we calculates a modified Pearson's Chi Squared statistic on
            history, and returns True (is random) if and only if the statistic
            is less than or equal to 3.
            """
            denom = turn - 2
            if internal_memory[2][6][0, 0] / denom >= 0.8:
                return False
            if (
                internal_memory[2][1] / denom < 0.25
                or internal_memory[2][1] / denom > 0.75
            ):
                return False
            if calculate_chi_squared(turn) > 3:
                return False
            return True
        def detect_streak(last_move):
            """
            Return true if and only if the opponent's last twenty moves are defects.
            """
            if last_move == 'D':
                internal_memory[2][10] += 1
            else:
                internal_memory[2][10] = 0
            if internal_memory[2][10] >= 20:
                return True
            return False
        def detect_parity_streak(last_move):
            """
            Switch which `parity_streak` we're pointing to and incerement if the
            opponent's last move was a Defection.  Otherwise reset the flag.  Then
            return true if and only if the `parity_streak` is at least
            `parity_limit`.
            This is similar to detect_streak with alternating streaks, except that
            these streaks get incremented elsewhere as well.
            """
            internal_memory[2][12] = 1 - internal_memory[2][12]  
            if last_move == 'D':
                internal_memory[2][11][internal_memory[2][12]] += 1
            else:
                internal_memory[2][11][internal_memory[2][12]] = 0
            if internal_memory[2][11][internal_memory[2][12]] >= internal_memory[2][13]:
                return True
        """Actual strategy definition that determines player's action."""
        turn = (len(history) + 1)
        if turn == 1:
            return 'C'
        if internal_memory[2][0] == "Defect":
            if history[-1][opp_index] == 'D':
                internal_memory[2][2] += 1
            else:
                internal_memory[2][2] -= 3
            if internal_memory[2][2] >= 11:
                internal_memory[2][0] = "Normal"
                internal_memory[2][4] = True
                internal_memory[2][7] = 2
                return try_return(to_return='C', lower_flags=False)
            return try_return('D')
        if not internal_memory[2][4]:
            if turn > 2:
                if history[-1][opp_index] == 'D':
                    internal_memory[2][1] += 1
                history_col = 1 if history[-1][opp_index] == 'D' else 0
                history_row = 1 if history[-2][opp_index] == 'D' else 0
                if history[-2][self_index] == 'D':
                    history_row += 2
                internal_memory[2][6][history_row, history_col] += 1
            if turn % 15 == 0 and turn > 15:
                if detect_random(turn):
                    internal_memory[2][0] = "Defect"
                    return try_return(
                        'D', lower_flags=False
                    )  
        if internal_memory[2][8] == 2 and history[-1][opp_index] == 'D':
            internal_memory[2][9] = True
        if (
            turn == 38
            and history[-1][opp_index] == 'D'
            and internal_memory[2][15] == 36
        ):
            internal_memory[2][0] = "Fair-weather"
            return try_return(to_return='C', lower_flags=False)
        if internal_memory[2][0] == "Fair-weather":
            if history[-1][opp_index] == 'D':
                internal_memory[2][0] = "Normal"  
            else:
                return try_return('C')
        if detect_streak(history[-1][opp_index]):
            return try_return('D', inc_parity=True)
        if detect_parity_streak(history[-1][opp_index]):
            internal_memory[2][11][internal_memory[2][12]] = (
                0  
            )
            internal_memory[2][14] += (
                1  
            )
            if internal_memory[2][14] >= 8:  
                internal_memory[2][13] = 3
            return try_return(
                'C', inc_parity=True
            )  
        if internal_memory[2][7] >= 1:
            return try_return('C', lower_flags=True, inc_parity=True)
        if turn < 37:
            return try_return(history[-1][opp_index], inc_parity=True)
        if turn == 37:
            internal_memory[2][7], internal_memory[2][8] = 2, 1
            return try_return('D', lower_flags=False)
        if internal_memory[2][9] or rnd.random() > internal_memory[2][5]:
            return try_return(history[-1][opp_index], inc_parity=True)
        internal_memory[2][5] += 0.05
        internal_memory[2][7], internal_memory[2][8] = 2, 1
        return try_return('D', lower_flags=False)
    elif strategy == 'Generous Tit For Tat':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'D' and rnd.random() > 0.1:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Pavlov / Win-Stay, Lose-Shift':
        if not len(history):
            return 'C'
        return history[-1][self_index] if self_payoff_matrix[history[-1][self_index] + history[-1][opp_index]] >= self_avg_payoff_matrix else flip(history[-1][self_index])
    elif strategy == 'First By Named Withheld':
        data = (
            internal_memory[5]
            if isinstance(internal_memory[5], list)
            else [0.3, 0, 0]
        )
        if len(data) < 3:
            data = [0.3, 0, 0]
        P = data[0]
        if len(history) > 0 and (len(history) + 1) % 10 == 0:
            lookback = min(10, len(history))
            recent_moves = [history[-i][opp_index] for i in range(lookback, 0, -1)]
            coop_count = recent_moves.count("C")
            coop_rate = coop_count / lookback if lookback > 0 else 0.5
            if coop_rate >= 0.8:
                P = min(P + 0.1, 0.9)
            elif coop_rate <= 0.2:
                P = max(P - 0.1, 0.1)
            else:
                P += rnd.uniform(-0.1, 0.1)
                P = max(min(P, 0.9), 0.1)
        if len(history) >= 130:
            if data[1] < data[2]:
                P = max(P - 0.1, 0.1)
        data[0] = P
        internal_memory[5] = data
        if rnd.random() < P:
            return "C"
        return "D"
    elif strategy == 'First By Davis':
        if (len(history) + 1) > 10:
            if internal_memory[7] == 1:
                return 'D'
            elif history[-1][opp_index] == 'D':
                internal_memory[7] = 1
                return 'D'
            else:
                return 'C'
        else:
            if len(history) > 0 and history[-1][opp_index] == 'D':
                internal_memory[7] = 1
            return 'C'
    elif strategy == 'Always Cooperate / Cooperator / Dove / Saint':
        return 'C'
    elif strategy == 'Always Defect / Defector / Hawk / Shark':
        return 'D'
    elif strategy == 'Random':
        return rnd.choice(['C','D'])
    elif strategy == 'First By Graaskamp / Second By Mauk':
        alpha = 0.05
        if (len(history) + 1) == 1:
            internal_memory[134]["d_count"] = 0
            internal_memory[134]["is_retaliating"] = False
            return 'C'
        if history[-1][opp_index] == 'D':
            internal_memory[134]["d_count"] += 1
        if internal_memory[134]["is_retaliating"]:
            return 'D'
        if (len(history) + 1) == 51:
            return 'D'
        if (len(history) + 1) > 50 and chisquare([internal_memory[134]["d_count"], len(history) - internal_memory[134]["d_count"]]).pvalue >= alpha:
            internal_memory[134]["is_retaliating"] = True
            return 'D'
        return history[-1][opp_index]
    elif strategy == '2 Tits For Tat':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'D' or internal_memory[10] > 0:
            if internal_memory[10] == 0:
                internal_memory[10] = 2
            elif internal_memory[10] > 0 and history[-1][opp_index] == 'D':
                internal_memory[10] += 1
            internal_memory[10] -= 1
            return 'D'
        else:
            return 'C'
    elif strategy == 'First By Stein & Rapoport':
        data = internal_memory[14][0] if isinstance(internal_memory[14], list) else internal_memory[14]
        if len(history) >= tournament_avg_last_round - 1:
            return 'D'
        if (len(history) + 1) > 2:
            prev_self = 0 if history[-1 - 1][self_index] == 'C' else 1
            prev_opp  = 0 if history[-1][opp_index] == 'C' else 1
            data["matrix"][prev_self, prev_opp] += 1
        if (len(history) + 1) % 15 == 0 and data["mode"] == "TFT":
            O = data["matrix"]
            row_sums = O.sum(axis=1)
            col_sums = O.sum(axis=0)
            total = O.sum()
            if total > 0 and np.all(row_sums > 0) and np.all(col_sums > 0):
                E = np.outer(row_sums, col_sums) / total
                chi2_stat = np.sum(((O - E) ** 2) / (E + 1e-9))
                if chi2_stat < 3.841:
                    data["mode"] = "ALLD"
        if data["mode"] == "ALLD":
            return 'D'
        opp_last = history[-1][opp_index] if len(history) > 0 else 'C'
        return opp_last
    elif strategy == 'First By Downing':
        if (len(history)) < 2:
            return 'D'
        if history[-1-1][self_index] == 'C':
            internal_memory[16][2] += 1
            if history[-1][opp_index] == 'C':
                internal_memory[16][0] += 1
        else:
            internal_memory[16][3] += 1
            if history[-1][opp_index] == 'C':
                internal_memory[16][1] += 1
        alpha = internal_memory[16][0] / internal_memory[16][2] if internal_memory[16][2] > 0 else 0.5
        beta = internal_memory[16][1] / internal_memory[16][3] if internal_memory[16][3] > 0 else 0.5
        E_c = (alpha * self_payoff_matrix['CC']) + ((1 - alpha) * self_payoff_matrix['CD'])
        E_d = (beta * self_payoff_matrix['DC']) + ((1 - beta) * self_payoff_matrix['DD'])
        if E_c > E_d:
            return 'C'
        if E_d > E_c:
            return 'D'
        return 'C' if history[-1][self_index] == 'D' else 'D'
    elif strategy == 'Suspicious Tit For Tat':
        if not len(history):
            return 'D'
        return history[-1][opp_index]
    elif strategy == 'First By Shubik':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'D' or internal_memory[33][0] > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[33][1] += 1
                if internal_memory[33][0] == 0:
                    internal_memory[33][0] = internal_memory[33][1]
            internal_memory[33][0] -= 1
            return 'D'
        else:
            return 'C'
    elif strategy == 'First By Nydegger':
        if not len(history):
            return 'C'
        elif (len(history) + 1) == 2:
            return history[-1][opp_index]
        elif (len(history) + 1) == 3:
            if (history[-1-1][opp_index] == 'D') and (history[-1][opp_index] == 'C'):
                return 'D'
            else:
                return history[-1][opp_index]
        else:
            black_list = (1, 6, 7, 17, 22, 23, 26, 29, 30, 31, 33, 38, 39, 45, 49, 54, 55, 58, 61)
            input_power = [1, 4, 16]
            def give_poin(x):
                opp = (history[-x][opp_index] == 'D') * 2
                self = (history[-x][self_index] == 'D')
                return (opp + self)
            inp = [give_poin(i) for i in range(1, 4)]
            A = sum(inp[i] * input_power[i] for i in range(3))
            if A in black_list:
                return 'D'
            else:
                return 'C'
    elif strategy == 'First By Grofman':
        if not len(history):
            return 'C'
        else:
            if history[-1][opp_index] == history[-1][self_index]:
                return 'C'
            else:
                if rnd.random() <= (2/7):
                    return 'C'
                else:
                    return 'D'
    elif strategy == 'Second By Champion':
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[81] += 1
            p_d = internal_memory[81] / (len(history))
        if (len(history)) < 10:
            return 'C'
        if (len(history)) < 25:
            return history[-1][opp_index]
        if history[-1][opp_index] == 'D':
            r = rnd.random()
            if p_d >= max(0.4, r):
                return 'D'
        return 'C'
    elif strategy == 'First By Tideman & Chieruzzi':
        def _decrease_retaliation_counter():
            """Lower the remaining owed retaliation count and flip to non-retaliate
            if the count drops to zero."""
            if internal_memory[82][0]:
                internal_memory[82][2] -= 1
                if internal_memory[82][2] == 0:
                    internal_memory[82][0] = False
        def _fresh_start():
            """Give the opponent a fresh start by forgetting the past"""
            internal_memory[82][0] = False
            internal_memory[82][1] = 0
            internal_memory[82][2] = 0
            internal_memory[82][5] = 0
        """Actual strategy definition that determines player's action."""
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'D':
            internal_memory[82][5] += 1
        if internal_memory[82][4]:
            internal_memory[82][4] = False
            return 'C'  
        current_round = (len(history) + 1)
        if internal_memory[82][3] == 0:
            valid_fresh_start = True
        else:
            valid_fresh_start = current_round - internal_memory[82][3] >= 20
        if valid_fresh_start:
            valid_points = score[self_index] - score[opp_index] >= 10
            valid_rounds = tournament_avg_last_round - current_round >= 10
            opponent_is_cooperating = history[-1][opp_index] == 'C'
            if valid_points and valid_rounds and opponent_is_cooperating:
                N = len(history)
                std_deviation = (N ** (1 / 2)) / 2
                lower = N / 2 - 3 * std_deviation
                upper = N / 2 + 3 * std_deviation
                if (
                    internal_memory[82][5] <= lower
                    or internal_memory[82][5] >= upper
                ):
                    internal_memory[82][3] = current_round
                    _fresh_start()
                    internal_memory[82][4] = True
                    return 'C'  
        if internal_memory[82][0]:
            _decrease_retaliation_counter()
            return 'D'
        if history[-1][opp_index] == 'D':
            internal_memory[82][0] = True
            internal_memory[82][1] += 1
            internal_memory[82][2] = internal_memory[82][1]
            _decrease_retaliation_counter()
            return 'D'
        return 'C'
    elif strategy == 'First By Tullock':
        if (len(history) + 1) < 11:
            return 'C'
        inp = [(history[-i][opp_index] == 'C') for i in range(10, 0, -1)] 
        t_c = sum(inp)
        p_c = (t_c / 10)
        return rnd_prob(max(0, p_c - 0.1))
    elif strategy == 'First By Feld':
        if len(history) > 0 and history[-1][opp_index] == 'D':
            return 'D'
        else:
            cooperate_chance = 1 - (((len(history)) / (tournament_avg_last_round)) * 0.5)
            if rnd.random() <= cooperate_chance:
                return 'C'
            else:
                return 'D'
    elif strategy == 'Second By White':
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[83] += 1
        if (len(history)) < 10:
            return 'C'
        if history[-1][opp_index] == 'C':
            return 'C'
        else:
            A = mth.floor(np.log((len(history) + 1))) * internal_memory[83]
            if A >= (len(history) + 1):
                return 'D'
            else:
                return 'C'
    elif strategy == 'Second By Black':
        prob_coop = {0: 1.0, 1: 1.0, 2: 0.88, 3: 0.68, 4: 0.4, 5: 0.04}
        if (len(history)) < 5:
            return 'C'
        inp = [(history[-i][opp_index] == 'D') for i in range(5, 0, -1)]
        return rnd_prob(prob_coop[sum(inp)])
    elif strategy == 'Second By Tranquilizer':
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[89][1] += 1
            else:
                internal_memory[89][1] = 0
            if internal_memory[89][0] == 2:
                internal_memory[89][0] = 0
                internal_memory[89][3] = (
                    (
                        internal_memory[89][3]
                        * internal_memory[89][5]
                    )
                    + (3 - (3 * internal_memory[89][6][history[-1][opp_index]]))
                    + (2 * internal_memory[89][6][history[-1][self_index]])
                    - (
                        (
                            internal_memory[89][6][history[-1][opp_index]]
                            * internal_memory[89][6][history[-1][self_index]]
                        )
                    )
                ) / (internal_memory[89][5] + 1)
                internal_memory[89][5] += 1
            elif internal_memory[89][0] == 1:
                internal_memory[89][0] = 2
                internal_memory[89][2] = (
                    (
                        internal_memory[89][2]
                        * internal_memory[89][4]
                    )
                    + (3 - (3 * internal_memory[89][6][history[-1][opp_index]]))
                    + (2 * internal_memory[89][6][history[-1][self_index]])
                    - (
                        (
                            internal_memory[89][6][history[-1][opp_index]]
                            * internal_memory[89][6][history[-1][self_index]]
                        )
                    )
                ) / (internal_memory[89][4] + 1)
                internal_memory[89][4] += 1
        else:
            return 'C'
        if history[-1][opp_index] == 'C':
            internal_memory[89][7] += 1
        if internal_memory[89][0] in (1, 2):
            return 'C'
        if (score[self_index] / (len(history) + 1)) >= (self_avg_payoff_matrix):
            probability = (
                0.95
                - (
                    (
                        internal_memory[89][2]
                        + internal_memory[89][3]
                        - 5
                    )
                    / 15
                )
                + (1 / ((len(history) + 1) * (len(history) + 1)))
                - (internal_memory[89][6][history[-1][opp_index]] / 4)
            )
            if rnd.random() <= probability:
                return 'C'
            internal_memory[89][0] += 1
            return 'D'
        if (score[self_index] / (len(history) + 1)) >= 1.75:
            probability = (
                (
                    0.25
                    + ((internal_memory[89][7] + 1) / (len(history) + 1))
                )
                - (internal_memory[89][1] * 0.25)
                + ((score[self_index] - score[opp_index]) / 100)
                + (4 / (len(history) + 1))
            )
            if rnd.random() <= probability:
                return 'C'
            return 'D'
        return history[-1][opp_index]
    elif strategy == 'Fool Me Once':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'D':
            internal_memory[88] += 1
        if internal_memory[88] > 1:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Gradual':
        if not len(history):
            return 'C'
        if internal_memory[101][2] > 0:
            internal_memory[101][2] -= 1
            return 'C'
        if history[-1][opp_index] == 'D' or internal_memory[101][0] > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[101][1] += 1
                if internal_memory[101][0] == 0:
                    internal_memory[101][0] = internal_memory[101][1]
            internal_memory[101][0] -= 1
            if internal_memory[101][0] == 0:
                internal_memory[101][2] = 2
            return 'D'
        else:
            return 'C'
    elif strategy == 'Adaptive Tit For Tat':
        omega = 0.5
        if len(history) > 0:
            if history[-1][opp_index] == 'C':
                A = internal_memory[104] + (omega * (1 - internal_memory[104]))
            else:
                A = internal_memory[104] - (omega * internal_memory[104])
        else:
            A = internal_memory[104]
        internal_memory[104] = A
        if A >= 0.5:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Second By Weiner':
        def try_return(to_return):
            if sum(internal_memory[35][3]) >= 5:
                return 'D'
            return to_return
        if (len(history) + 1) >= 3:
            internal_memory[35][3][internal_memory[35][4]] = (history[-2][opp_index] == 'D')
            internal_memory[35][4] = (internal_memory[35][4] + 1) % 12
        if internal_memory[35][0]:
           internal_memory[35][0] = False
           internal_memory[35][2] = 0
           if (
               internal_memory[35][1] < (len(history) + 1)
               and history[-1][opp_index] == 'D'
           ):
               internal_memory[35][1] += 20
               return try_return('C')
           else:
               return try_return(history[-1][opp_index])
        else:
            if (history[-1][opp_index] == 'D' if len(history) > 0 else False):
                internal_memory[35][2] += 1
            else:
                if internal_memory[35][2] % 2 == 1:
                    internal_memory[35][0] = True
                internal_memory[35][2] = 0
            return try_return(history[-1][opp_index] if len(history) > 0 else 'C')
    elif strategy == 'Easy Go / Fool Me Forever / Reverse Grudger':
        if len(history) > 0:
            if (internal_memory[64] == 1) or (history[-1][opp_index] == 'D'):
                if internal_memory[64] == 0:
                    internal_memory[64] = 1
                return 'C'
        return 'D'
    elif strategy == 'Second By Borufsen':
        def try_return(to_return):
            if to_return == 'C':
                return 'C'
            if internal_memory[121][4]:
                internal_memory[121][4] = False
                return 'C'
            return 'D'
        if (len(history)) < 2:
            return 'C'
        if (len(history) + 1) > 3:
            if history[-1][opp_index] == 'C':
                if history[-2][self_index] == 'C':
                    internal_memory[121][1] += 1
                else:
                    internal_memory[121][0] += 1
        if (len(history) + 1) > 2 and (len(history) + 1) % 25 == 2:
            coming_from_defect = False
            if internal_memory[121][5] == "Defect":
                coming_from_defect = True
            internal_memory[121][5] = "Normal"
            coops = internal_memory[121][0] + internal_memory[121][1]
            if coops < 3:
                internal_memory[121][5] = "Defect"
            if (8 <= coops <= 17) and internal_memory[121][1] / coops < 0.7:
                internal_memory[121][5] = "Defect"
            internal_memory[121][0], internal_memory[121][1] = 0, 0
            if internal_memory[121][5] == "Defect":
                internal_memory[121][2] = 0
                internal_memory[121][3] = 0
                internal_memory[121][4] = False
            if internal_memory[121][5] == "Normal" and coming_from_defect:
                return 'D'
        if internal_memory[121][5] == "Defect":
            return 'D'
        else:
            assert internal_memory[121][5] == "Normal", "What do you mean?"
            if (history[-1][self_index], history[-1][opp_index]) == ('D', 'D'):
                internal_memory[121][2] += 1
            else:
                internal_memory[121][2] = 0
            if internal_memory[121][2] >= 3:
                internal_memory[121][2] = 0
                internal_memory[121][3] = 0
                return try_return('C')
            my_two_back, opp_two_back = 'C', 'C'
            if (len(history) + 1) >= 3:
                my_two_back = history[-2][self_index]
                opp_two_back = history[-2][opp_index]
            if (
                history[-1][self_index] != history[-1][opp_index]
                and history[-1][self_index] == opp_two_back
                and history[-1][opp_index] == my_two_back
            ):
                internal_memory[121][3] += 1
            else:
                internal_memory[121][3] = 0
            if internal_memory[121][3] >= 3:
                internal_memory[121][2] = 0
                internal_memory[121][3] = 0
                internal_memory[121][4] = True
            return try_return(history[-1][opp_index])
    elif strategy == 'Second By Graaskamp & Katzen':
        if not len(history):
            return 'C'
        if (len(history) % 10) == 0:
            A = -10 + ((len(history) + 1) * self_payoff_matrix['CC'])
            if score[self_index] < A:
                internal_memory[122] = 1
        if internal_memory[122] == 0:
            return history[-1][opp_index]
        else:
            return 'D'
    elif strategy == 'Worse & Worse 3':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'C':
            internal_memory[123] += 1
        p_c = internal_memory[123] / (len(history))
        if rnd.random() <= p_c:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Second By Wiliam Adams':
        if not len(history):
            return 'C'
        if (len(history) + 1) == 2:
            internal_memory[124][1] = (history[-1][opp_index] == 'D')
        if history[-1][opp_index] == 'D':
            internal_memory[124][0] += 1
        number_of_defection = internal_memory[124][0] - internal_memory[124][1]
        if number_of_defection <= 9:
            if internal_memory[124][0] in (4, 7, 9):
                return 'D'
            else:
                return 'C'
        else:
            if history[-1][opp_index] == 'D':
                A = 1 - (0.5 ** (number_of_defection - 9))
                if rnd.random() <= A:
                    return 'D'
            return 'C'
    elif strategy == 'Second By Yamachi':
        def try_return(to_return, opp_def):
            """
            Return `to_return`, unless the turn is greater than 40 AND
            `portion_defect` is between 45% and 55%.
            In this case, still record the history as `to_return` so that the
            modified behavior doesn't affect the calculation of `count_us_them_us`.
            """
            turn = (len(history) + 1)
            internal_memory[125][1].append(to_return)
            if turn > 40:
                portion_defect = (opp_def + 0.5) / turn
                if 0.45 < portion_defect < 0.55:
                    return 'D'
            return to_return
        turn = (len(history) + 1)
        if turn == 1:
            return try_return('C', 0)
        if history[-1][opp_index] == 'D': internal_memory[125][2] += 1
        us_last = internal_memory[125][1][-1]
        them_two_ago, us_two_ago, them_three_ago = 'C', 'C', 'C'
        if turn >= 3:
            them_two_ago = history[-2][opp_index]
            us_two_ago = history[-2][self_index]
        if turn >= 4:
            them_three_ago = history[-3][opp_index]
        if turn >= 3:
            internal_memory[125][0][
                (them_three_ago, us_two_ago, them_two_ago)
            ] += 1
        if (
            internal_memory[125][0][(them_two_ago, us_last, 'C')]
            >= internal_memory[125][0][(them_two_ago, us_last, 'D')]
        ):
            return try_return('C', internal_memory[125][2])
        return try_return('D', internal_memory[125][2])
    elif strategy == 'Prober':
        if (len(history)) < 3:
            return ('D', 'C', 'C')[(len(history) + 1)-1]
        if (history[1][opp_index] + history[2][opp_index]) == 'CC':
            return 'D'
        else:
            return history[-1][opp_index]
    elif strategy == 'Contrite Tit For Tat':
        """Actual strategy definition that determines player's action."""
        if not len(history):
            internal_memory[131][1].append('C')
            return 'C'
        if internal_memory[131][0] and history[-1][self_index] == 'C':
            internal_memory[131][0] = False
            internal_memory[131][1].append('C')
            return 'C'
        if internal_memory[131][1][-1] != history[-1][self_index]:  
            if history[-1][self_index] == 'D' and history[-1][opp_index] == 'C':
                internal_memory[131][0] = True
        internal_memory[131][1].append(history[-1][opp_index])
        return history[-1][opp_index]
    elif strategy == 'Cycle Hunter':
        if not len(history):
            return 'C'
        if cycle_detec(history, opp_index, min_ = 3):
            return 'D'
        return 'C'
    elif strategy == 'Evolved Looker Up 2_2_2':
        if (len(history) + 1) > 2:
            inp = (history[-2][self_index] + history[-1][self_index], history[-2][opp_index] + history[-1][opp_index], history[0][opp_index] + history[1][opp_index])
            lookup_table = {('CC', 'CC', 'CC'): 'C', ('CC', 'CC', 'CD'): 'D', ('CC', 'CC', 'DC'): 'D', ('CC', 'CC', 'DD'): 'C', ('CC', 'CD', 'CC'): 'D', ('CC', 'CD', 'CD'): 'C', ('CC', 'CD', 'DC'): 'D', ('CC', 'CD', 'DD'): 'D', ('CC', 'DC', 'CC'): 'C', ('CC', 'DC', 'CD'): 'D', ('CC', 'DC', 'DC'): 'D', ('CC', 'DC', 'DD'): 'D', ('CC', 'DD', 'CC'): 'C', ('CC', 'DD', 'CD'): 'D', ('CC', 'DD', 'DC'): 'D', ('CC', 'DD', 'DD'): 'D', ('CD', 'CC', 'CC'): 'D', ('CD', 'CC', 'CD'): 'D', ('CD', 'CC', 'DC'): 'C', ('CD', 'CC', 'DD'): 'D', ('CD', 'CD', 'CC'): 'C', ('CD', 'CD', 'CD'): 'D', ('CD', 'CD', 'DC'): 'C', ('CD', 'CD', 'DD'): 'C', ('CD', 'DC', 'CC'): 'C', ('CD', 'DC', 'CD'): 'D', ('CD', 'DC', 'DC'): 'D', ('CD', 'DC', 'DD'): 'C', ('CD', 'DD', 'CC'): 'C', ('CD', 'DD', 'CD'): 'D', ('CD', 'DD', 'DC'): 'C', ('CD', 'DD', 'DD'): 'D', ('DC', 'CC', 'CC'): 'D', ('DC', 'CC', 'CD'): 'D', ('DC', 'CC', 'DC'): 'C', ('DC', 'CC', 'DD'): 'C', ('DC', 'CD', 'CC'): 'C', ('DC', 'CD', 'CD'): 'C', ('DC', 'CD', 'DC'): 'C', ('DC', 'CD', 'DD'): 'D', ('DC', 'DC', 'CC'): 'D', ('DC', 'DC', 'CD'): 'D', ('DC', 'DC', 'DC'): 'C', ('DC', 'DC', 'DD'): 'D', ('DC', 'DD', 'CC'): 'D', ('DC', 'DD', 'CD'): 'D', ('DC', 'DD', 'DC'): 'D', ('DC', 'DD', 'DD'): 'D', ('DD', 'CC', 'CC'): 'D', ('DD', 'CC', 'CD'): 'D', ('DD', 'CC', 'DC'): 'D', ('DD', 'CC', 'DD'): 'D', ('DD', 'CD', 'CC'): 'C', ('DD', 'CD', 'CD'): 'C', ('DD', 'CD', 'DC'): 'D', ('DD', 'CD', 'DD'): 'D', ('DD', 'DC', 'CC'): 'C', ('DD', 'DC', 'CD'): 'D', ('DD', 'DC', 'DC'): 'D', ('DD', 'DC', 'DD'): 'D', ('DD', 'DD', 'CC'): 'C', ('DD', 'DD', 'CD'): 'C', ('DD', 'DD', 'DC'): 'C', ('DD', 'DD', 'DD'): 'D'}
            return lookup_table[inp]
        return 'C'
    elif strategy == 'Alternator / Second By Duisman':
        if (len(history) % 2) == 0:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Suspicious Alternator':
        if (len(history) % 2) == 0:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Sneaky Tit For Tat':
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[142] = True
        if len(history) < 2:
            return 'C'
        if not internal_memory[142]:
            return 'D'
        if (history[-1][opp_index], history[-2][self_index]) == ('D', 'D'):
            return 'C'
        return history[-1][opp_index]
    elif strategy == 'Meta Majority':
        meta_majo_team = [
            call_strategy('First By Tit For Tat', history, opp_index, opp_name, None, None, None, None, None, None),
            call_strategy('Always Cooperate / Cooperator / Dove / Saint', history, opp_index, opp_name, None, None, None, None, None, None),
            call_strategy('Always Defect / Defector / Hawk / Shark', history, opp_index, opp_name, None, None, None, None, None, None),
            call_strategy('Random', history, opp_index, opp_name, None, None, None, None, None, None),
            call_strategy('First By Grudger / Grim Trigger', history, opp_index, opp_name, internal_memory[424], None, None, None, None, None),
            call_strategy('Pavlov / Win-Stay, Lose-Shift', history, opp_index, opp_name, None, None, None, None, None, None),
            call_strategy('Generous Tit For Tat', history, opp_index, opp_name, None, None, None, None, None, None),
            call_strategy('Sample / Second By Tit For 2 Tats', history, opp_index, opp_name, None, None, None, None, None, None),
        ]
        meta_majo_majority = sum((1 if i == 'C' else -1) for i in meta_majo_team)
        if meta_majo_majority < 0:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Meta Winner':
        payoff = {('C', 'C'): self_payoff_matrix['CC'],('C', 'D'): self_payoff_matrix['CD'],('D', 'C'): self_payoff_matrix['DC'],('D', 'D'): self_payoff_matrix['DD']}
        meta_win_team = [
            call_strategy('First By Tit For Tat', history, opp_index, opp_name, None, None, None, None, None, None),
            call_strategy('Always Cooperate / Cooperator / Dove / Saint', history, opp_index, opp_name, None, None, None, None, None, None),
            call_strategy('Always Defect / Defector / Hawk / Shark', history, opp_index, opp_name, None, None, None, None, None, None),
            call_strategy('Random', history, opp_index, opp_name, None, None, None, None, None, None),
            call_strategy('First By Grudger / Grim Trigger', history, opp_index, opp_name, internal_memory[136][1], None, None, None, None, None),
            call_strategy('Pavlov / Win-Stay, Lose-Shift', history, opp_index, opp_name, None, None, None, None, None, None),
            call_strategy('Generous Tit For Tat', history, opp_index, opp_name, None, None, None, None, None, None),
            call_strategy('Sample / Second By Tit For 2 Tats', history, opp_index, opp_name, None, None, None, None, None, None),
        ]
        if len(history) > 0:
            internal_memory[136][0] = [internal_memory[136][0][i] + payoff[(internal_memory[136][2][i], history[-1][opp_index])] for i in range(len(internal_memory[136][0]))]
        else:
            internal_memory[136][0] = [0 for _ in range(len(meta_win_team))]
        internal_memory[136][2] = meta_win_team[:]
        best_score = max(internal_memory[136][0])
        meta_win_majority = sum((1 if meta_win_team[i] == 'C' else -1) for i in range(len(meta_win_team)) if internal_memory[136][0][i] == best_score)
        if meta_win_majority < 0:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Risky Q-Learner':
        brain = internal_memory[139][0]
        another_factor = internal_memory[139][1]
        A = min(12, len(history)) - 1
        opp_inp = tuple(['C' for _ in range(12 - min(12, len(history)))] + [history[-i][opp_index] for i in range(min(12, len(history)), 0, -1)]) 
        self_inp = tuple(['C' for _ in range(12 - min(12, len(history)))] + [history[-i][self_index] for i in range(min(12, len(history)), 0, -1)]) 
        inp = (opp_inp, self_inp)
        if brain.get(inp) == None:
            brain[inp] = [0,0]
        if len(history) > 0 and another_factor[0][1] != 0.5:
            reward = 0
            if history[-1][opp_index] == 'C':
                if history[-1][self_index] == 'C':
                    reward = self_payoff_matrix['CC']
                else:
                    reward = self_payoff_matrix['DC']
            else:
                if history[-1][self_index] == 'C':
                    reward = self_payoff_matrix['CD']
                else:
                    reward = self_payoff_matrix['DD']
            c_factor = brain[inp][0]
            d_factor = brain[inp][1]
            a = 0.9
            gamma = 0.9
            Q_old = brain[another_factor[0][0]][another_factor[0][1]]
            Q_new = Q_old + (a * (reward + ((gamma * max(c_factor, d_factor)) - Q_old)))
            brain[another_factor[0][0]][another_factor[0][1]] = Q_new
        if rnd.random() <= another_factor[1]:
            Action = rnd_prob(0.5)
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        decay = 1
        min_epsilon = 0
        another_factor[1] = max(another_factor[1] * decay, min_epsilon)
        if brain[inp][0] == brain[inp][1]:
            Action = rnd_prob(0.5)
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        best_choice = max(brain[inp])
        if best_choice == brain[inp][0]:
            another_factor[0] = [inp, 0]
            return 'C'
        else:
            another_factor[0] = [inp, 1]
            return 'D'
    elif strategy == 'Arrogant Q-Learner':
        brain = internal_memory[140][0]
        another_factor = internal_memory[140][1]
        A = min(12, len(history)) - 1
        opp_inp = tuple(['C' for _ in range(12 - min(12, len(history)))] + [history[-i][opp_index] for i in range(min(12, len(history)), 0, -1)]) 
        self_inp = tuple(['C' for _ in range(12 - min(12, len(history)))] + [history[-i][self_index] for i in range(min(12, len(history)), 0, -1)]) 
        inp = (opp_inp, self_inp)
        if brain.get(inp) == None:
            brain[inp] = [0,0]
        if len(history) > 0 and another_factor[0][1] != 0.5:
            reward = 0
            if history[-1][opp_index] == 'C':
                if history[-1][self_index] == 'C':
                    reward = self_payoff_matrix['CC']
                else:
                    reward = self_payoff_matrix['DC']
            else:
                if history[-1][self_index] == 'C':
                    reward = self_payoff_matrix['CD']
                else:
                    reward = self_payoff_matrix['DD']
            c_factor = brain[inp][0]
            d_factor = brain[inp][1]
            a = 0.9
            gamma = 0.1
            Q_old = brain[another_factor[0][0]][another_factor[0][1]]
            Q_new = Q_old + (a * (reward + ((gamma * max(c_factor, d_factor)) - Q_old)))
            brain[another_factor[0][0]][another_factor[0][1]] = Q_new
        if rnd.random() <= another_factor[1]:
            Action = rnd_prob(0.5)
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        decay = 1
        min_epsilon = 0
        another_factor[1] = max(another_factor[1] * decay, min_epsilon)
        if brain[inp][0] == brain[inp][1]:
            Action = rnd_prob(0.5)
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        best_choice = max(brain[inp])
        if best_choice == brain[inp][0]:
            another_factor[0] = [inp, 0]
            return 'C'
        else:
            another_factor[0] = [inp, 1]
            return 'D'
    elif strategy == 'Hesitant Q-Learner':
        brain = internal_memory[141][0]
        another_factor = internal_memory[141][1]
        A = min(12, len(history)) - 1
        opp_inp = tuple(['C' for _ in range(12 - min(12, len(history)))] + [history[-i][opp_index] for i in range(min(12, len(history)), 0, -1)]) 
        self_inp = tuple(['C' for _ in range(12 - min(12, len(history)))] + [history[-i][self_index] for i in range(min(12, len(history)), 0, -1)]) 
        inp = (opp_inp, self_inp)
        if brain.get(inp) == None:
            brain[inp] = [0,0]
        if len(history) > 0 and another_factor[0][1] != 0.5:
            reward = 0
            if history[-1][opp_index] == 'C':
                if history[-1][self_index] == 'C':
                    reward = self_payoff_matrix['CC']
                else:
                    reward = self_payoff_matrix['DC']
            else:
                if history[-1][self_index] == 'C':
                    reward = self_payoff_matrix['CD']
                else:
                    reward = self_payoff_matrix['DD']
            c_factor = brain[inp][0]
            d_factor = brain[inp][1]
            a = 0.1
            gamma = 0.9
            Q_old = brain[another_factor[0][0]][another_factor[0][1]]
            Q_new = Q_old + (a * (reward + ((gamma * max(c_factor, d_factor)) - Q_old)))
            brain[another_factor[0][0]][another_factor[0][1]] = Q_new
        if rnd.random() <= another_factor[1]:
            Action = rnd_prob(0.5)
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        decay = 1
        min_epsilon = 0
        another_factor[1] = max(another_factor[1] * decay, min_epsilon)
        if brain[inp][0] == brain[inp][1]:
            Action = rnd_prob(0.5)
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        best_choice = max(brain[inp])
        if best_choice == brain[inp][0]:
            another_factor[0] = [inp, 0]
            return 'C'
        else:
            another_factor[0] = [inp, 1]
            return 'D'
    elif strategy == 'Cautious Q-Learner':
        brain = internal_memory[300][0]
        another_factor = internal_memory[300][1]
        A = min(12, len(history)) - 1
        opp_inp = tuple(['C' for _ in range(12 - min(12, len(history)))] + [history[-i][opp_index] for i in range(min(12, len(history)), 0, -1)]) 
        self_inp = tuple(['C' for _ in range(12 - min(12, len(history)))] + [history[-i][self_index] for i in range(min(12, len(history)), 0, -1)]) 
        inp = (opp_inp, self_inp)
        if brain.get(inp) == None:
            brain[inp] = [0,0]
        if len(history) > 0 and another_factor[0][1] != 0.5:
            reward = 0
            if history[-1][opp_index] == 'C':
                if history[-1][self_index] == 'C':
                    reward = self_payoff_matrix['CC']
                else:
                    reward = self_payoff_matrix['DC']
            else:
                if history[-1][self_index] == 'C':
                    reward = self_payoff_matrix['CD']
                else:
                    reward = self_payoff_matrix['DD']
            c_factor = brain[inp][0]
            d_factor = brain[inp][1]
            a = 0.1
            gamma = 0.1
            Q_old = brain[another_factor[0][0]][another_factor[0][1]]
            Q_new = Q_old + (a * (reward + ((gamma * max(c_factor, d_factor)) - Q_old)))
            brain[another_factor[0][0]][another_factor[0][1]] = Q_new
        if rnd.random() <= another_factor[1]:
            Action = rnd_prob(0.5)
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        decay = 1
        min_epsilon = 0
        another_factor[1] = max(another_factor[1] * decay, min_epsilon)
        if brain[inp][0] == brain[inp][1]:
            Action = rnd_prob(0.5)
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        best_choice = max(brain[inp])
        if best_choice == brain[inp][0]:
            another_factor[0] = [inp, 0]
            return 'C'
        else:
            another_factor[0] = [inp, 1]
            return 'D'
    elif strategy == 'Thue Morse Inverse':
        if not len(history):
            return 'C'
        else:
            if internal_memory[143][1] == 0:
                thuemorse_code = internal_memory[143][0]
                for i in internal_memory[143][0]:
                    if i == '0':
                        thuemorse_code += '1'
                    else:
                        thuemorse_code += '0'
                internal_memory[143][0] = thuemorse_code
                internal_memory[143][1] = len(thuemorse_code)
            internal_memory[143][1] -= 1
            return 'D' if internal_memory[143][0][len(internal_memory[143][0]) - (internal_memory[143][1] + 1)] == '1' else 'C'
    elif strategy == 'Cycler CCD':
        return ('C', 'C', 'D')[(len(history)) % 3]
    elif strategy == 'Bully / Reverse Tit For Tat':
        if not len(history):
            return 'D'
        return 'D' if history[-1][opp_index] == 'C' else 'C'
    elif strategy == 'Anti Tit For Tat / Psycho':
        if not len(history):
            return 'C'
        else:
            return 'D' if history[-1][opp_index] == 'C' else 'C'
    elif strategy == 'Evolved Looker Up 1_1_1':
        if len(history) > 0:
            inp = (history[-1][self_index], history[-1][opp_index], history[0][opp_index])
            lookup_table = {('C', 'C', 'C'): 'C', ('C', 'C', 'D'): 'D', ('C', 'D', 'C'): 'D', ('C', 'D', 'D'): 'D', ('D', 'C', 'C'): 'D', ('D', 'C', 'D'): 'C', ('D', 'D', 'C'): 'D', ('D', 'D', 'D'): 'D'}
            return lookup_table[inp]
        return 'C'
    elif strategy == 'Tricky Cooperator':
        """Almost always cooperates, but will try to trick the opponent by
        defecting.
        Defect once in a while in order to get a better payout.
        After 3 rounds, if opponent has not defected to a max history depth of
        10, defect.
        """
        def _has_played_enough_rounds_to_be_tricky():
            return (len(history)) >= internal_memory[145][0]
        def _opponents_has_cooperated_enough_to_be_tricky():
            rounds_to_be_checked = [history[-i][opp_index] for i in range(1, min(-internal_memory[145][0], default_memory_looked) + 1)] 
            return 'D' not in rounds_to_be_checked
        if (
            _has_played_enough_rounds_to_be_tricky()
            and _opponents_has_cooperated_enough_to_be_tricky()
        ):
            return 'D'
        return 'C'
    elif strategy == 'Second By Cave':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'D':
            internal_memory[148] += 1
        p_d = internal_memory[148] / (len(history) + 1)
        if (len(history) + 1) > 39:
            if p_d > 0.39:
                return 'D'
        if (len(history) + 1) > 29:
            if p_d > 0.65:
                return 'D'
        if (len(history) + 1) > 19:
            if p_d > 0.79:
                return 'D'
        if history[-1][opp_index] == 'D':
            if internal_memory[148] > 17:
                return 'D'
            else:
                return rnd_prob(0.5)
        else:
            return 'C'
    elif strategy == 'Second By Kluepfel':
        if not len(history):
            return 'C'
        if (len(history) + 1) > 2:
            if history[-2][self_index] == 'D':
                if history[-1][opp_index] == 'C':
                    internal_memory[149][0] += 1 
                else:
                    internal_memory[149][1] += 1 
            else:
                if history[-1][opp_index] == 'C':
                    internal_memory[149][2] += 1 
                else:
                    internal_memory[149][3] += 1 
        if (len(history) + 1) > 27:
            if internal_memory[149][0] >= (
                internal_memory[149][0] + internal_memory[149][1]
            ) / 2 - 0.75 * np.sqrt(
                internal_memory[149][0] + internal_memory[149][1]
            ) and internal_memory[149][3] >= (
                internal_memory[149][3] + internal_memory[149][2]
            ) / 2 - 0.75 * np.sqrt(
                internal_memory[149][3] + internal_memory[149][2]
            ):
                return 'D'
        kluepfel_eyes = [('C', 'C') for _ in range(3)] + history
        inp = [kluepfel_eyes[-i][opp_index] for i in range(1, 3 + 1)] 
        if inp[0] == inp[1] and inp[1] == inp[2]:
            return inp[0]
        A = rnd.random()
        if inp[0] == inp[1]:
            if A < 0.9:
                return inp[0]
            else:
                return 'C' if inp[0] == 'D' else 'D'
        if inp[0] == 'C':
            if A < 0.7:
                return inp[0]
            else:
                return 'C' if inp[0] == 'D' else 'D'
        if inp[0] == 'D':
            if A < 0.6:
                return inp[0]
            else:
                return 'C' if inp[0] == 'D' else 'D'
    elif strategy == 'Second By Getzler':
        if not len(history): return 'C'
        if history[-1][opp_index] == 'D':
            internal_memory[150] += 1
        internal_memory[150] *= 0.5
        if rnd.random() < internal_memory[150]:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Second By Leyvraz':
        prob_coop = {
            ('C', 'C', 'C'): 1.0,
            ('C', 'C', 'D'): 0.5,  
            ('C', 'D', 'C'): 0.0,  
            ('C', 'D', 'D'): 0.25,  
            ('D', 'C', 'C'): 1.0,  
            ('D', 'C', 'D'): 1.0,  
            ('D', 'D', 'C'): 1.0,  
            ('D', 'D', 'D'): 0.25,  
        }
        B = min(3, len(history))
        inp = tuple((['C'] * (3 - B)) + [history[-i][opp_index] for i in range(B, 0, -1)])
        return rnd_prob(prob_coop[inp])
    elif strategy == 'Second By Eatherley':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'D':
            internal_memory[151] += 1
        else:
            return 'C'
        p_d = internal_memory[151] / (len(history))
        return rnd_prob(1 - p_d)
    elif strategy == 'Second By Richard Hufford':
        turn = (len(history) + 1)
        if turn == 1:
            return 'C'
        internal_memory[84][2] = (internal_memory[84][2] + 1) % 4
        me_two_moves_ago = 'C'
        if turn > 2:
            me_two_moves_ago = history[-2][self_index]
        if me_two_moves_ago == history[-1][opp_index]:
            internal_memory[84][0] += 1
            internal_memory[84][1][internal_memory[84][2]] = 1
        else:
            internal_memory[84][1][internal_memory[84][2]] = 0
        if turn < internal_memory[84][5]:
            if history[-1][opp_index] == 'C':
                internal_memory[84][4] += 1
            else:
                internal_memory[84][4] = 0
            if internal_memory[84][4] >= internal_memory[84][3]:
                internal_memory[84][5] = turn
                if internal_memory[84][4] == internal_memory[84][3]:
                    return 'D'
        elif turn == internal_memory[84][5] + 2:
            if history[-1][opp_index] == 'C':
                internal_memory[84][6] += 1
            else:
                internal_memory[84][7] += 1
            internal_memory[84][3] = (
                np.floor(
                    20.0 * internal_memory[84][7] / internal_memory[84][6]
                )
                + 1
            )
            internal_memory[84][4] = 0
            return 'C'
        proportion_agree = internal_memory[84][0] / turn
        last_four_num = sum(internal_memory[84][1])
        if proportion_agree > 0.9 and last_four_num >= 4:
            return 'C'
        elif proportion_agree >= 0.625 and last_four_num >= 2:
            return history[-1][opp_index]
        return 'D'
    elif strategy == 'Second By Colbert':
        if (len(history)) < 5:
            return 'C'
        elif (len(history) + 1) == 6:
            return 'D'
        if internal_memory[152] > 0:
            A = ('D', 'C', 'C')[3 - internal_memory[152]]
            internal_memory[152] -= 1
            return A
        if history[-1][opp_index] == 'D':
            internal_memory[152] = 3
            return 'D'
        return 'C'
    elif strategy == 'Second By Mikkelson':
        turn = (len(history) + 1)
        if turn == 1:
            return 'C'
        if history[-1][opp_index] == 'C':
            internal_memory[154][0] += 1
            if internal_memory[154][0] > 8:
                internal_memory[154][0] = 8
        else:
            internal_memory[154][0] -= 2
            if internal_memory[154][0] < -7:
                internal_memory[154][0] = -7
            internal_memory[154][1] += 1
        if turn == 2:
            return 'C'
        if internal_memory[154][0] > 0:
            return 'C'
        if turn <= 10:
            internal_memory[154][0] = 4
            return 'D'
        if internal_memory[154][1] / turn >= 0.15:
            return 'D'
        return 'C'
    elif strategy == 'Second By Rowsam':
        turn = (len(history) + 1)
        if internal_memory[155][0] == "Defect":
            return 'D'
        if internal_memory[155][0] == "Coop Def Cycle 1":
            internal_memory[155][0] = "Coop Def Cycle 2"
            return 'C'
        if internal_memory[155][0] == "Coop Def Cycle 2":
            internal_memory[155][0] = "Normal"
            return 'D'
        if turn % 18 == 0:
            if internal_memory[155][1] >= 3:
                internal_memory[155][1] -= 1
        if turn % 6 != 0:
            return 'C'
        points_per_turn = score[self_index] / turn  
        if points_per_turn < 1.0:
            internal_memory[155][1] += 5
        elif points_per_turn < 1.5:
            internal_memory[155][1] += 3
        elif points_per_turn < 2.0:
            internal_memory[155][1] += 2
        elif points_per_turn < 2.5:
            internal_memory[155][1] += 1
        else:
            return 'C'
        if internal_memory[155][1] >= 7:
            internal_memory[155][0] = "Defect"
        else:
            internal_memory[155][0] = "Coop Def Cycle 1"
        return 'D'
    elif strategy == 'Second By Appold':
        turn = (len(history) + 1)
        us_two_turns_ago = 'C' if turn <= 2 else history[-2][self_index]
        if turn > 1:
            internal_memory[156][1][us_two_turns_ago] += 1
        if turn > 1 and history[-1][opp_index] == 'C':
            internal_memory[156][0][us_two_turns_ago] += 1
        if turn <= 4:
            return 'C'
        if history[-1][opp_index] == 'D' and not internal_memory[156][2]:
            internal_memory[156][2] = True
            return 'C'
        prob_coop = (
            internal_memory[156][0][us_two_turns_ago]
            / internal_memory[156][1][us_two_turns_ago]
        )
        return 'C' if rnd.random() <= prob_coop else 'D'
    elif strategy == 'Second By Almy':
        k39r = (history[-1][self_index] == 'D') if len(history) > 0 else 0
        if k39r + ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 2:
            internal_memory[129]["bothd"] += 1
        if k39r + ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) < 2:
            internal_memory[129]["bothd"] = 0
        internal_memory[129]["count"] -= 1
        k39r = 0
        voldmv = internal_memory[129]["oldmov"]
        internal_memory[129]["oldmov"] = ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
        if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 1:
            internal_memory[129]["tatcnt"] += 1
        if internal_memory[129]["evil"] == 0 and ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 1:
            internal_memory[129]["evil"] = 1
        while True:
            step = internal_memory[129]["step"]
            substp = internal_memory[129]["substp"]
            if step == 1:
                if substp == 1:
                    internal_memory[129]["count"] = 10
                    internal_memory[129]["tatcnt"] = 0
                    internal_memory[129]["titcnt"] = 0
                    internal_memory[129]["substp"] = 2
                    continue
                elif substp == 2:
                    if voldmv + internal_memory[129]["oldmov"] == 2:
                        k39r = 1
                    internal_memory[129]["titcnt"] += k39r
                    if internal_memory[129]["count"] == 0:
                        internal_memory[129]["substp"] = 3
                    return 'CD'[k39r]
                elif substp == 3:
                    oldstp = internal_memory[129]["step"]
                    internal_memory[129]["ok"][internal_memory[129]["step"]] = score[self_index] - internal_memory[129]["totk"]
                    internal_memory[129]["totk"] = score[self_index]
                    internal_memory[129]["substp"] = 1
                    if internal_memory[129]["tatcnt"] > 0:
                        internal_memory[129]["step"] = 1
                        for i1 in range(1, 3):
                            for i2 in range(2, 4):
                                if (
                                    internal_memory[129]["ok"][i1] == 0
                                    or internal_memory[129]["ok"][i2] == 0
                                ):
                                    continue
                                if internal_memory[129]["ok"][i1] >= internal_memory[129]["ok"][i2]:
                                    continue
                                if internal_memory[129]["step"] == i1:
                                    internal_memory[129]["step"] = i2
                        if internal_memory[129]["step"] != 3:
                            if (
                                internal_memory[129]["ok"][internal_memory[129]["step"] + 1] == 0
                                and (internal_memory[129]["tatcnt"] >= 4 or internal_memory[129]["titcnt"] == 0)
                            ):
                                internal_memory[129]["step"] += 1
                        if internal_memory[129]["step"] < oldstp and internal_memory[129]["bothd"] > 0:
                            internal_memory[129]["step"] = 5
                        continue
                    else:
                        internal_memory[129]["step"] = 4
                        if internal_memory[129]["evil"] == 1:
                            internal_memory[129]["step"] = 1
                        elif internal_memory[129]["evil"] == 0:
                            internal_memory[129]["evil"] = -1
                        continue
            elif step == 2:
                if substp == 1:
                    internal_memory[129]["count"] = 10
                    internal_memory[129]["tatcnt"] = 0
                    internal_memory[129]["titcnt"] = 0
                    internal_memory[129]["substp"] = 2
                    continue
                elif substp == 2:
                    if internal_memory[129]["oldmov"] == 1:
                        k39r = 1
                    internal_memory[129]["titcnt"] += k39r
                    if internal_memory[129]["count"] == 0:
                        internal_memory[129]["substp"] = 3
                    return 'CD'[k39r]
                elif substp == 3:
                    internal_memory[129]["substp"] = 1
                    internal_memory[129]["step"] = 1
                    continue
            elif step == 3:
                if substp == 1:
                    internal_memory[129]["count"] = 10
                    internal_memory[129]["tatcnt"] = 0
                    internal_memory[129]["titcnt"] = 0
                    internal_memory[129]["substp"] = 2
                    continue
                elif substp == 2:
                    k39r = 1
                    internal_memory[129]["titcnt"] += 1
                    if internal_memory[129]["count"] == 0:
                        internal_memory[129]["substp"] = 3
                    return 'CD'[k39r]
                elif substp == 3:
                    internal_memory[129]["substp"] = 1
                    internal_memory[129]["step"] = 1
                    continue
            elif step == 4:
                if substp == 1:
                    internal_memory[129]["substp"] = 2
                    k39r = 1
                    internal_memory[129]["count"] = internal_memory[129]["n"]
                    internal_memory[129]["tatcnt"] = 0
                    return 'CD'[k39r]
                elif substp == 2:
                    if internal_memory[129]["count"] == 0:
                        internal_memory[129]["substp"] = 3
                    return 'CD'[k39r]
                elif substp == 3:
                    if internal_memory[129]["tatcnt"] != 0:
                        if internal_memory[129]["f"] == 0:
                            internal_memory[129]["n"] += 1
                            internal_memory[129]["substp"] = 1
                            internal_memory[129]["step"] = 1
                            continue
                        else:
                            internal_memory[129]["substp"] = 4
                            if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 1:
                                internal_memory[129]["n"] += 1
                            internal_memory[129]["tatcnt"] = ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
                            return 'CD'[k39r]
                    else:
                        internal_memory[129]["f"] = 1
                        internal_memory[129]["substp"] = 1
                        continue
                elif substp == 4:
                    if internal_memory[129]["tatcnt"] <= 4:
                        return 'CD'[k39r]
                    internal_memory[129]["substp"] = 1
                    internal_memory[129]["step"] = 1
                    continue
            elif step == 5:
                if internal_memory[129]["substp"] != 2:
                    internal_memory[129]["count"] = 5
                    internal_memory[129]["substp"] = 2
                if internal_memory[129]["count"] != 0:
                    return 'CD'[k39r]
                internal_memory[129]["substp"] = 1
                internal_memory[129]["step"] = 1
                continue
    elif strategy == 'Second By Ambuehl & Hickey':
        if not len(history):
            return 'C'
        if (len(history) + 1) < 6:
            return history[-1][opp_index]
        return choice_majority([history[-i][opp_index] for i in range(5, 0, -1)], history[-5][opp_index])
    elif strategy == 'Second By Pinkley':
        if (len(history) + 1) == 1:
            return 'C'
        if (len(history) + 1) > 2:
            n_idx = internal_memory[158]["n"]
            if 0 <= n_idx < 4:
                if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 0:
                    internal_memory[158]["qc"][n_idx] += 1.0
                internal_memory[158]["qn"][n_idx] += 1.0
            if n_idx == 0:
                internal_memory[158]["x"] = internal_memory[158]["qc"][0] / internal_memory[158]["qn"][0]
                internal_memory[158]["px"] = 1.0 - internal_memory[158]["x"]
            elif n_idx == 2:
                internal_memory[158]["y"] = internal_memory[158]["qc"][2] / internal_memory[158]["qn"][2]
                internal_memory[158]["py"] = 1.0 - internal_memory[158]["y"]
            elif n_idx == 1:
                internal_memory[158]["z"] = internal_memory[158]["qc"][1] / internal_memory[158]["qn"][1]
                internal_memory[158]["pz"] = 1.0 - internal_memory[158]["z"]
            elif n_idx == 3:
                internal_memory[158]["w"] = internal_memory[158]["qc"][3] / internal_memory[158]["qn"][3]
                internal_memory[158]["pw"] = 1.0 - internal_memory[158]["w"]
        x = internal_memory[158]["x"]
        px = internal_memory[158]["px"]
        y = internal_memory[158]["y"]
        py = internal_memory[158]["py"]
        z = internal_memory[158]["z"]
        pz = internal_memory[158]["pz"]
        w = internal_memory[158]["w"]
        pw = internal_memory[158]["pw"]
        e = [0.0] * 11
        e[0] = (3.0 * z) / (z + px) if (z + px) != 0 else 0.0
        e[1] = (3.0 * (y * z + w * pz) + 5.0 * z * px + px * pz) / (
            y * z + w * pz + px + z * px + px * pz
            if (y * z + w * pz + px + z * px + px * pz) != 0
            else 1.0
        )
        e[2] = (3.0 * w * y + 5.0 * w * px + px * pz) / (
            w * y + 2.0 * w * px + px * pz
            if (w * y + 2.0 * w * px + px * pz) != 0
            else 1.0
        )
        e[3] = (3.0 * w * py + 5.0 * z * px + px * py) / (
            w * py + px * py + z * px + px * py
            if (w * py + px * py + z * px + px * py) != 0
            else 1.0
        )
        e[4] = (3.0 * z + 5.0 * x * z + z * px) / (
            1.0 - x * y - w * px + 2.0 * z
            if (1.0 - x * y - w * px + 2.0 * z) != 0
            else 1.0
        )
        e[5] = (8.0 * w * z + z * px) / (
            2.0 * w * z + w * py + z * px if (2.0 * w * z + w * py + z * px) != 0 else 1.0
        )
        e[6] = (3.0 * z * py + 5.0 * x * z + z * py) / (
            2.0 * z * py + pw * py + x * z
            if (2.0 * z * py + pw * py + x * z) != 0
            else 1.0
        )
        e[7] = (3.0 * (y * z + w * pz) + 5.0 * (z * pw + w * x) + 1.0 - x * y - z * py) / (
            y * z + w * pz + 2.0 - 2.0 * x * y - w * px + z * pw + w * x - z * py
            if (
                y * z + w * pz + 2.0 - 2.0 * x * y - w * px + z * pw + w * x - z * py
            )
            != 0
            else 1.0
        )
        e[8] = (3.0 * w * y + 5.0 * w + 1.0 - x * y - z * py) / (
            2.0 * w + 1.0 - x * y - z * py
            if (2.0 * w + 1.0 - x * y - z * py) != 0
            else 1.0
        )
        e[9] = (3.0 * w * py + 5.0 * (z * pw + w * x) + py) / (
            py + z * pw + w * x + py if (py + z * pw + w * x + py) != 0 else 1.0
        )
        e[10] = (5.0 * w + py) / (w + py if (w + py) != 0 else 1.0)
        ibest = 0
        beste = e[0]
        for i in range(1, 11):
            if e[i] > beste:
                ibest = i
                beste = e[i]
        internal_memory[158]["n"] = (2 * internal_memory[158]["iold"] + ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)) % 4
        k91r = internal_memory[158]["ipol"][ibest][internal_memory[158]["n"]]
        internal_memory[158]["iold"] = k91r
        return 'CD'[k91r]
    elif strategy == 'Second By Pebly':
        r = rnd.random()
        k70r = (history[-1][self_index] == 'D') if len(history) > 0 else 0
        if internal_memory[159]["jz"] == ((history[-1][opp_index] == 'D') if len(history) > 0 else 0):
            k70r = internal_memory[159]["jz"]
        else:
            k70r = 0
            if r > 0.2:
                k70r = 1
        internal_memory[159]["jz"] = k70r
        return 'CD'[k70r]
    elif strategy == 'Second By Falk & Langsted':
        if (len(history) + 1) == 1:
            k85r = 0
            internal_memory[160]["f1"] = k85r
            return 'CD'[k85r]
        j5 = internal_memory[160]["j0"] / 1e07
        j3 = int(j5)
        internal_memory[160]["j8"] = (j5 - j3) * 1e07
        f5 = internal_memory[160]["f0"] / 1e07
        f3 = int(f5)
        internal_memory[160]["f8"] = (f5 - f3) * 1e07
        internal_memory[160]["j0"] = internal_memory[160]["j8"] * 10 + ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
        internal_memory[160]["f0"] = internal_memory[160]["f8"] * 10 + internal_memory[160]["f1"]
        if internal_memory[160]["f1"] != 0:
            if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 0:
                internal_memory[160]["i1"] += 1
            else:
                internal_memory[160]["i2"] += 1
        else:
            if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 0:
                internal_memory[160]["i3"] += 1
            else:
                internal_memory[160]["i4"] += 1
        if (len(history) + 1) > 20:
            i5_val = internal_memory[160]["i1"] + 1e-6
            i6_val = internal_memory[160]["i2"] + 1e-6
            x8_val = internal_memory[160]["i3"] + 1e-6
            i8_val = internal_memory[160]["i4"] + 1e-6
            a_val = i5_val / i6_val
            b_val = x8_val / i8_val
            if 0.5 <= a_val <= 1.5 and 0.5 <= b_val <= 1.5:
                try:
                    internal_memory[160]["f1"] = k85r
                except NameError:
                    internal_memory[160]["f1"] = 0
                k85r = 1
                internal_memory[160]["d"] = 0
                return 'CD'[k85r]
        if internal_memory[160]["t"] == 1:
            k85r = ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
            return 'CD'[k85r]
        if int(internal_memory[160]["j0"]) == 11111111:
            internal_memory[160]["t"] = 1
            k85r = ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
            return 'CD'[k85r]
        if internal_memory[160]["c"] == 1:
            internal_memory[160]["c"] = 0
            internal_memory[160]["f1"] = 1
            k85r = 0
            return 'CD'[k85r]
        z1 = internal_memory[160]["j0"] / 100.0
        z2 = int(z1)
        z3 = z1 - z2
        internal_memory[160]["j2"] = int(z3 * 100.0)
        if (len(history)) < 30 and internal_memory[160]["j2"] == 11:
            pass
        else:
            z4 = internal_memory[160]["j0"] / 10000.0
            z5 = int(z4)
            z6 = z4 - z5
            internal_memory[160]["j4"] = int(z6 * 10000.0)
            w8 = internal_memory[160]["f0"] / 10000.0
            z8 = int(w8)
            z9 = w8 - z8
            internal_memory[160]["f4"] = int(z9 * 10000.0)
            if internal_memory[160]["j4"] == 1011 and internal_memory[160]["f4"] == 111:
                internal_memory[160]["t"] = 1
                k85r = ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
                return 'CD'[k85r]
        y1 = internal_memory[160]["i2"] + internal_memory[160]["i4"]
        y2 = internal_memory[160]["i1"] + internal_memory[160]["i2"] + 3
        if y1 >= y2:
            internal_memory[160]["f1"] = 0
            k85r = 1
            internal_memory[160]["d"] = 0
            return 'CD'[k85r]
        if internal_memory[160]["d"] == 1:
            internal_memory[160]["d"] = 0
            internal_memory[160]["f1"] = 1
            k85r = 0
            return 'CD'[k85r]
        internal_memory[160]["f1"] = 0
        k85r = 0
        return 'CD'[k85r]
    elif strategy == 'Second By Weiderman':
        k38r = (history[-1][self_index] == 'D') if len(history) > 0 else 0
        if internal_memory[161][1] == 1:
            return 'D'
        if internal_memory[161][0] >= 4:
            internal_memory[161][0] -= 4
        internal_memory[161][0] = internal_memory[161][0] * 2 + ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
        if internal_memory[161][0] == 7:
            internal_memory[161][1] = 1
        k38r = internal_memory[161][1]
        return 'CD'[k38r]
    elif strategy == 'Second By Dawes & Batell':
        if (len(history) + 1) == 1:
            return 'C'
        if internal_memory[162]["mode"] == 1:
            return 'D'
        if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 1:
            internal_memory[162]["inod"] += 1
            inoc = (len(history) + 1) - internal_memory[162]["inod"]
            t1 = 1.6667 ** internal_memory[162]["inod"]
            t2 = 0.882 ** inoc
            test_val = t1 * t2
            if test_val >= 5.0:
                internal_memory[162]["mode"] = 1
                return 'D'
        return 'C'
    elif strategy == 'Second By Lefevre':
        if len(history) > 0 and history[-1][opp_index] == 'D':
            internal_memory[26] += 1
        k37r = (history[-1][self_index] == 'D') if len(history) > 0 else 0
        if 5 * internal_memory[26] > (len(history) + 1):
            k37r = 1
        return 'CD'[k37r]
    elif strategy == 'Second By Quayle':
        k56r = (history[-1][self_index] == 'D') if len(history) > 0 else 0
        if (len(history) + 1) >= 3:
            if internal_memory[168]["past"] == 1:
                if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 0:
                    internal_memory[168]["nice2"] += 1
                internal_memory[168]["totdef"] += 1
                internal_memory[168]["bad"] = float(internal_memory[168]["nice2"]) / float(internal_memory[168]["totdef"])
            else:
                if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 0:
                    internal_memory[168]["nice1"] += 1
                internal_memory[168]["totcop"] += 1
                internal_memory[168]["good"] = float(internal_memory[168]["nice1"]) / float(internal_memory[168]["totcop"])
        internal_memory[168]["past"] = k56r
        good = internal_memory[168]["good"]
        bad = internal_memory[168]["bad"]
        c = 6.0 * good - 8.0 * bad - 2.0
        alt = 4.0 * good - 5.0 * bad - 1.0
        if c >= 0.0 and c >= alt:
            k56r = 0
        elif c >= 0.0 and c < alt:
            k56r = 1 - k56r
        elif alt >= 0.0:
            k56r = 1 - k56r
        else:
            k56r = 1
        internal_memory[168]["past"] = k56r
        return 'CD'[k56r]
    elif strategy == 'Second By Anderson':
        if (len(history) + 1) == 1:
            return 'CD'[internal_memory[195]["iold1"]]
        if (len(history) + 1) >= 3:
            if internal_memory[195]["iold2"] == 1:
                internal_memory[195]["ndc"] += 1 - ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
                internal_memory[195]["ndd"] += 1 * ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
            else:
                internal_memory[195]["ncc"] += 1 - ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
                internal_memory[195]["ncd"] += 1 * ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
        internal_memory[195]["iold2"] = internal_memory[195]["iold1"]
        if (len(history) + 1) < 16:
            if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 0:
                internal_memory[195]["iold1"] = 0
                return 'CD'[internal_memory[195]["iold1"]]
            if internal_memory[195]["kount"] >= 3:
                internal_memory[195]["iold1"] = 0
                return 'CD'[internal_memory[195]["iold1"]]
            internal_memory[195]["kount"] += 1
            internal_memory[195]["iold1"] = 1
            return 'CD'[internal_memory[195]["iold1"]]
        if (len(history) + 1) == 17 and ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 1 and internal_memory[195]["ncd"] == 1 and internal_memory[195]["ndd"] == 0:
            internal_memory[195]["mytwin"] = 1
        if (internal_memory[195]["ncd"] * 3) >= (internal_memory[195]["ncc"] + internal_memory[195]["ncd"]):
            internal_memory[195]["iold1"] = 1
            return 'CD'[internal_memory[195]["iold1"]]
        if (len(history) + 1) % 4 != 0:
            internal_memory[195]["iold1"] = 0
            return 'CD'[internal_memory[195]["iold1"]]
        if internal_memory[195]["mytwin"] == 1:
            internal_memory[195]["iold1"] = 0
            return 'CD'[internal_memory[195]["iold1"]]
        if internal_memory[195]["ndc"] >= ((len(history) + 1) // 12) or internal_memory[195]["ndd"] == 0:
            internal_memory[195]["iold1"] = 1
            return 'CD'[internal_memory[195]["iold1"]]
        internal_memory[195]["iold1"] = 0
        return 'CD'[internal_memory[195]["iold1"]]
    elif strategy == 'Second By Zimmerman':
        k73r = internal_memory[388]["ipost"]
        if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) != internal_memory[388]["itest"]:
            return 'CD'[k73r]
        if internal_memory[388]["itest"] == 1:
            internal_memory[388]["idunu"] += 1
        elif internal_memory[388]["itest"] == 0:
            internal_memory[388]["idunb"] += 1
        if internal_memory[388]["idunu"] < internal_memory[388]["iaggd"] and internal_memory[388]["idunb"] < internal_memory[388]["ipayb"]:
            return 'CD'[k73r]
        internal_memory[388]["idunu"] = 0
        internal_memory[388]["idunb"] = 0
        internal_memory[388]["ipost"] = 1 if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 1 else 0
        k73r = internal_memory[388]["ipost"]
        internal_memory[388]["itest"] = 1 if internal_memory[388]["ipost"] == 0 else 0
        if internal_memory[388]["itest"] != 0:
            rata_rata_skor = score[self_index] // (len(history) + 1)
            internal_memory[388]["iaggd"] = internal_memory[388]["iaggd"] - 3 + rata_rata_skor
            if internal_memory[388]["iaggd"] <= 0:
                internal_memory[388]["iaggd"] = 1
        else:
            internal_memory[388]["ipayb"] = int(1.6667 * float(internal_memory[388]["iaggd"] + 1))
        return 'CD'[k73r]
    elif strategy == 'Second By Newman':
        k55r = (history[-1][self_index] == 'D') if len(history) > 0 else 0
        if (len(history) + 1) == 1:
            k55r = 0
            internal_memory[163]["iold"] = k55r
            return 'CD'[k55r]
        if (len(history) + 1) > 2:
            if internal_memory[163]["iold"] == 0:
                if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 0:
                    internal_memory[163]["qca"] += 1
                internal_memory[163]["qna"] += 1
                if internal_memory[163]["qna"] > 0:
                    internal_memory[163]["alpha"] = float(internal_memory[163]["qca"]) / float(internal_memory[163]["qna"])
            else:
                if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 0:
                    internal_memory[163]["qcb"] += 1
                internal_memory[163]["qnb"] += 1
                if internal_memory[163]["qnb"] > 0:
                    internal_memory[163]["beta"] = float(internal_memory[163]["qcb"]) / float(internal_memory[163]["qnb"])
        internal_memory[163]["iold"] = k55r
        alpha = internal_memory[163]["alpha"]
        beta = internal_memory[163]["beta"]
        polc = 6.0 * alpha - 9.0 * beta - 2.0
        polalt = 4.0 * alpha - 6.0 * beta - 1.0
        if polc >= 0:
            if polc >= polalt:
                k55r = 0
                internal_memory[163]["iold"] = k55r
                return 'CD'[k55r]
            else:
                k55r = 1 - k55r
                internal_memory[163]["iold"] = k55r
                return 'CD'[k55r]
        else:
            if polalt >= 0:
                k55r = 1 - k55r
                internal_memory[163]["iold"] = k55r
                return 'CD'[k55r]
            else:
                k55r = 1
                if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 0 or internal_memory[163]["iold"] == 0:
                    internal_memory[163]["mutdef"] = 0
                else:
                    internal_memory[163]["mutdef"] += 1
                    if internal_memory[163]["mutdef"] > 3:
                        k55r = 0
                internal_memory[163]["iold"] = k55r
                return 'CD'[k55r]
    elif strategy == 'Second By Jones':
        def _jones_sub_evaluation(l4, k, l):
            x = [0.0] * 8
            d4 = internal_memory[197]["t4"]
            if d4 > 7:
                d4 -= 8
            a1 = l4[d4][0]
            a2 = l4[d4][1]
            if a2 == 0:
                a2 = 1.0
            a3 = a1 / a2
            a = 3.0 * a3
            b = a + a3 + 1.0
            for c in range(4):
                x[c] = a
                x[c + 4] = b
            t4_local = internal_memory[197]["t4"]
            if t4_local > 4:
                t4_local -= 4
            t4_local *= 2
            for c in range(8):
                d4_sub = t4_local
                if c in [4, 5, 6, 7]:
                    d4_sub += 1
                if d4_sub == 9:
                    d4_sub = 1
                if d4_sub > 7:
                    d4_sub -= 8
                a1 = l4[d4_sub][0]
                a2 = l4[d4_sub][1]
                if a2 == 0:
                    a2 = 1.0
                a3 = a1 / a2
                a_val = 3.0 * a3
                b_val = a_val + a3 + 1.0
                if c in [2, 3, 6, 7]:
                    x[c] += b_val
                else:
                    x[c] += a_val
            max_val = -999.0
            s1 = 1
            for c in range(8):
                if x[c] > max_val:
                    max_val = x[c]
                    s1 = c + 1
            internal_memory[197]["s1"] = s1
            if s1 < 5:
                return 0
            else:
                return 1
        l4 = internal_memory[197]["l4"]
        if (len(history) + 1) == 81 and score[self_index] == score[opp_index] and score[self_index] == 237:
            internal_memory[197]["t0"] = 1
        if (len(history) + 1) == 1:
            k81r = 0
        else:
            if (len(history) + 1) == 2 and ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 1:
                internal_memory[197]["t9"] = 9
            if (len(history) + 1) < internal_memory[197]["t9"]:
                k81r = ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
            else:
                if internal_memory[197]["t5"] > 7:
                    internal_memory[197]["t5"] -= 8
                if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 0:
                    l4[internal_memory[197]["t5"]][0] += 1.0
                if internal_memory[197]["t9"] == 9 and internal_memory[197]["t0"] == 1:
                    if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 1:
                        internal_memory[197]["t0"] = 0
                        k81r = _jones_sub_evaluation(
                            l4, score[self_index], score[opp_index]
                        )
                    else:
                        t2 = 0
                        k81r = 0
                        temp_m = (len(history) + 1)
                        matched = False
                        while temp_m >= 140 + t2:
                            if (
                                80 + t2 < (len(history) + 1) < 140 + t2
                            ):
                                k81r = 1
                                matched = True
                                break
                            elif (
                                140 + t2 <= (len(history)) < 180 + t2
                            ):
                                k81r = 0
                                matched = True
                                break
                            t2 += 100
                        if not matched:
                            k81r = _jones_sub_evaluation(l4, score[self_index], score[opp_index])
                else:
                    if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 1:
                        internal_memory[197]["t8"] += 1
                        if 8 <= internal_memory[197]["t8"] <= 9:
                            k81r = 0
                        else:
                            if internal_memory[197]["t8"] > 1:
                                internal_memory[197]["t8"] = 1
                            k81r = _jones_sub_evaluation(l4, score[self_index], score[opp_index])
                    else:
                        if 0 <= internal_memory[197]["t8"] < 6:
                            internal_memory[197]["t8"] = 0
                            k81r = _jones_sub_evaluation(l4, score[self_index], score[opp_index])
                        else:
                            if internal_memory[197]["t8"] > 0:
                                internal_memory[197]["t8"] = -200
                            k81r = 0
                            internal_memory[197]["t8"] += 1
        internal_memory[197]["t5"] = internal_memory[197]["t4"]
        if (len(history) + 1) % 10 == 0:
            for c in range(8):
                l4[c][0] *= 9.0
            internal_memory[197]["t6"] += 1
        if internal_memory[197]["t4"] > 7:
            internal_memory[197]["t4"] -= 8
        if (len(history) + 1) > 3:
            l4[internal_memory[197]["t4"]][1] += 1.0
        if internal_memory[197]["t4"] > 4:
            internal_memory[197]["t4"] -= 4
        internal_memory[197]["t4"] = (internal_memory[197]["t4"] * 2) + k81r
        return 'CD'[k81r]
    elif strategy == 'Second By Schurmann':
        r = rnd.random()
        if (len(history) + 1) == 1:
            return 'C'
        internal_memory[153]["s"] = 2 * ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) + internal_memory[153]["h"] + 1
        if internal_memory[153]["z"] == 1:
            if internal_memory[153]["s"] > 1:
                if internal_memory[153]["s"] == 4:
                    internal_memory[153]["q6"] = 0.74 * internal_memory[153]["q6"] + 0.104
                else:
                    internal_memory[153]["q6"] = 0.5 * internal_memory[153]["q6"]
            else:
                internal_memory[153]["q6"] = internal_memory[153]["q6"] * 0.57 + 0.43
            k87r = 1
            internal_memory[153]["h"] = 1
            if r > internal_memory[153]["q6"]:
                return 'CD'[k87r]
            else:
                k87r = 0
                internal_memory[153]["h"] = 0
                return 'CD'[k87r]
        else:
            if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 0:
                k87r = 0
                internal_memory[153]["h"] = 0
                return 'CD'[k87r]
            else:
                internal_memory[153]["z"] = 1
                if internal_memory[153]["s"] > 1:
                    if internal_memory[153]["s"] == 4:
                        internal_memory[153]["q6"] = 0.74 * internal_memory[153]["q6"] + 0.104
                    else:
                        internal_memory[153]["q6"] = 0.5 * internal_memory[153]["q6"]
                else:
                    internal_memory[153]["q6"] = internal_memory[153]["q6"] * 0.57 + 0.43
                k87r = 1
                internal_memory[153]["h"] = 1
                if r > internal_memory[153]["q6"]:
                    return 'CD'[k87r]
                else:
                    k87r = 0
                    internal_memory[153]["h"] = 0
                    return 'CD'[k87r]
    elif strategy == 'Second By Nussbacher':
        if (len(history)) < 10:
            return 'C'
        r = rnd.random()
        d = sum((history[-i][opp_index] == 'D') for i in range(10, 0, -1))
        if d > 8.9:
            if r < 0.94:
                return 'D'
            return 'C'
        elif d == 8 or d == 4 or d == 3:
            if r < 0.915:
                return 'D'
            return 'C'
        elif d == 7 or d == 6 or d == 5 or d == 2:
            if r < 0.87:
                return 'D'
            return 'C'
        elif d == 1:
            if r < 0.23:
                return 'D'
            return 'C'
        else:
            return 'C'
    elif strategy == 'Second By Batell':
        k65r = (history[-1][self_index] == 'D') if len(history) > 0 else 0
        if (len(history) + 1) == 1:
            return 'C'
        if internal_memory[164]["totd"] >= 10:
            return 'D'
        if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 0:
            return 'C'
        internal_memory[164]["totd"] += 1
        if internal_memory[164]["totd"] >= 10:
            return 'D'
        if internal_memory[164]["lastd"] == 0:
            internal_memory[164]["lastd"] = (len(history) + 1)
            return 'C'
        internal_memory[164]["diff"] = (len(history) + 1) - internal_memory[164]["lastd"]
        if internal_memory[164]["diff"] <= 4:
            internal_memory[164]["totd"] = 10
            return 'D'
        internal_memory[164]["lastd"] = (len(history) + 1)
        return 'C'
    elif strategy == 'Second By Smith':
        r = rnd.random()
        k52r = 0
        internal_memory[165]["d9"] += 1
        if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 0:
            internal_memory[165]["d9"] = 0
        if internal_memory[165]["d9"] >= 2:
            k52r = 1
            if internal_memory[165]["d9"] < (5 + 3 * internal_memory[165]["d8"]):
                pass
            else:
                internal_memory[165]["d9"] = 0
                internal_memory[165]["d8"] += 1
        if r <= 0.05:
            k52r = 1 - k52r
        return 'CD'[k52r]
    elif strategy == 'Second By Leyland':
        r = rnd.random()
        if (len(history) + 1) == 1:
            return 'C'
        k82r = ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
        internal_memory[166]["i5"] += ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
        internal_memory[166]["d4"] += ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
        if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 0 and internal_memory[166]["i5"] > 1:
            if internal_memory[166]["i5"] > 5:
                internal_memory[166]["i5"] = 0
                internal_memory[166]["i1"] = 0
                internal_memory[166]["i3"] = 1 if internal_memory[166]["i5"] <= 5 else 0
                internal_memory[166]["i5"] = 0
                internal_memory[166]["i1"] = 0
                return 'C'
            else:
                internal_memory[166]["i5"] = 0
        if (len(history) + 1) >= 30:
            if internal_memory[166]["i3"] != 0:
                rasio_defeksi = internal_memory[166]["d4"] / len(history)
                if abs(rasio_defeksi - 0.5) < 0.1:
                    internal_memory[166]["x"] -= 0.2
                if internal_memory[166]["i2"] == 1:
                    if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 0:
                        internal_memory[166]["x"] -= 0.05
                        if internal_memory[166]["x"] < 0.0:
                            internal_memory[166]["x"] = 0.0
                        internal_memory[166]["i2"] = 0
                        if internal_memory[166]["x"] < 0.3:
                            if r > internal_memory[166]["x"]:
                                k82r = 1
                                internal_memory[166]["i1"] = 1
                                return 'CD'[k82r]
                            internal_memory[166]["i2"] = internal_memory[166]["i1"]
                            return 'CD'[k82r]
                        return 'CD'[k82r]
                    else:
                        internal_memory[166]["x"] += 0.15
                        if internal_memory[166]["x"] > 1.0:
                            internal_memory[166]["x"] = 1.0
                        internal_memory[166]["i2"] = 0
                        internal_memory[166]["i3"] = 1
                        if internal_memory[166]["i5"] > 5:
                            internal_memory[166]["i3"] = 0
                        internal_memory[166]["i5"] = 0
                        internal_memory[166]["i1"] = 0
                        return 'CD'[k82r]
                else:
                    if r > internal_memory[166]["x"]:
                        k82r = 1
                        internal_memory[166]["i1"] = 1
                        return 'CD'[k82r]
                    internal_memory[166]["i2"] = internal_memory[166]["i1"]
        return 'CD'[k82r]
    elif strategy == 'Second By Mcgurrin':
        k45r = (history[-1][self_index] == 'D') if len(history) > 0 else 0
        if (len(history) + 1) == 1:
            return 'D'
        if (len(history) + 1) == 2:
            if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 1:
                internal_memory[380]["d"] = 1
            else:
                internal_memory[380]["d"] = 0
            return 'C'
        if (len(history) + 1) == 3:
            k45r = 0
            if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 1:
                if internal_memory[380]["d"] == 1:
                    internal_memory[380]["c"] = 1
            else:
                if internal_memory[380]["d"] != 1:
                    internal_memory[380]["a"] = 1
            return 'CD'[k45r]
        if internal_memory[380]["c"] == 1:
            k45r = ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
            return 'CD'[k45r]
        if internal_memory[380]["b"] == 1:
            k45r = 0
            if internal_memory[380]["jold"] == 1 and ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 1:
                k45r = 1
            internal_memory[380]["jold"] = ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
            return 'CD'[k45r]
        if internal_memory[380]["a"] == 1:
            k45r = 1
            internal_memory[380]["e"] += 1
            if internal_memory[380]["e"] != 8:
                if not (internal_memory[380]["jold"] == 1 and ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 1):
                    k45r = 0
            else:
                internal_memory[380]["e"] = 0
            internal_memory[380]["jold"] = ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
            return 'CD'[k45r]
        if internal_memory[380]["d"] == 1:
            if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 1:
                k45r = 1
                internal_memory[380]["c"] = 1
            else:
                k45r = 0
                internal_memory[380]["b"] = 1
            return 'CD'[k45r]
        if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 1:
            k45r = 0
            internal_memory[380]["c"] = 1
        else:
            k45r = 0
            internal_memory[380]["b"] = 1
        return 'CD'[k45r]
    elif strategy == 'Second By Hollander':
        r = rnd.random()
        k62r = 0
        if (len(history) + 1) == internal_memory[167]["iran"]:
            k62r = 1
            internal_memory[167]["iran"] = int(23 * r + (len(history) + 1) + 1)
        else:
            if internal_memory[167]["jold"] == 1 and ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 1:
                k62r = 1
        internal_memory[167]["jold"] = ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
        return 'CD'[k62r]
    elif strategy == 'Second By Smoody':
        r = rnd.random()
        k50r = 0
        if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 0 and r >= 0.9:
            k50r = 1
        return 'CD'[k50r]
    elif strategy == 'Second By Snodgrass':
        sc = internal_memory[169]["sc"]
        sl = internal_memory[169]["sl"]
        st = internal_memory[169]["st"]
        gt = internal_memory[169]["gt"]
        tm = internal_memory[169]["tm"]
        while True:
            code = internal_memory[169]["cn"] // 10
            if code * 10 == internal_memory[169]["cn"] and 1 <= code <= 6:
                if code - 1 < len(sc):
                    sc[code - 1] = score[self_index]
            if code - 1 < len(sl) and sl[code - 1] == 1:
                internal_memory[169]["cn"] += 1
                tm[code - 1] += 1
                if code == 1:
                    return 'C'
                elif code == 2:
                    return 'D'
                elif code == 3:
                    k89r_val = 1 - internal_memory[169]["mylm"]
                    internal_memory[169]["mylm"] = k89r_val
                    return 'CD'[k89r_val]
                elif code == 4:
                    return history[-1][opp_index]
                elif code == 5:
                    k89r_val = (
                        1 if (((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 1 and internal_memory[169]["hlm"] == 1) else 0
                    )
                    internal_memory[169]["hlm"] = ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
                    return 'CD'[k89r_val]
                elif code == 6:
                    sgt = 0
                    for i in range(5):
                        st[i] = sc[i + 1] - sc[i]
                        sgt += st[i]
                        gt[i] += st[i]
                    mean = sgt // internal_memory[169]["csrc"] if internal_memory[169]["csrc"] > 0 else 0
                    amean = 9 * mean // 10
                    internal_memory[169]["csrc"] = 0
                    for i in range(5):
                        if sl[i] == 1:
                            if st[i] < amean:
                                sl[i] = 0
                        else:
                            if tm[i] > 0 and (10 * gt[i] // tm[i]) > amean:
                                sl[i] = 1
                        if sl[i] == 1:
                            internal_memory[169]["csrc"] += 1
                    internal_memory[169]["cn"] = 10
                    continue
            else:
                internal_memory[169]["cn"] += 10
                if internal_memory[169]["cn"] > 60:
                    internal_memory[169]["cn"] = 10
                continue
            break
        return 'C'
    elif strategy == 'Second By Robertson':
        k54r = (history[-1][self_index] == 'D') if len(history) > 0 else 0
        if (len(history) + 1) == 1:
            return 'C'
        if (len(history) + 1) == 20:
            internal_memory[171]["dl"] = 0.10
        if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 0:
            internal_memory[171]["stdef"] = 0
            internal_memory[171]["coops"] += 1
            if float(internal_memory[171]["opdef"]) > float((len(history) + 1)) * internal_memory[171]["dl"]:
                pass
            elif (len(history) + 1) % internal_memory[171]["nd"] == 0 and internal_memory[171]["okdef"]:
                k54r = 1
                internal_memory[171]["mydef"] = False
                internal_memory[171]["nodef"] += 1
                if internal_memory[171]["nodef"] % 6 == 0:
                    internal_memory[171]["nd"] -= 1
                if internal_memory[171]["nd"] < 1:
                    internal_memory[171]["nd"] = 1
                return 'CD'[k54r]
            else:
                internal_memory[171]["mydef"] = False
                return 'C'
        internal_memory[171]["coops"] = 0
        if (len(history)) < 4:
            k54r = 1
            return 'CD'[k54r]
        internal_memory[171]["stdef"] += 1
        internal_memory[171]["opdef"] += 1
        if internal_memory[171]["mydef"]:
            internal_memory[171]["okdef"] = False
        if float(internal_memory[171]["opdef"]) > float((len(history) + 1)) * internal_memory[171]["dl"]:
            pass
        elif internal_memory[171]["stdef"] > 2:
            pass
        else:
            internal_memory[171]["mydef"] = False
            return 'C'
        if 20 * internal_memory[171]["opdef"] <= internal_memory[171]["coops"] * (len(history) + 1):
            return 'C'
        k54r = 1
        internal_memory[171]["mydef"] = False
        return 'CD'[k54r]
    elif strategy == 'Second By Rabbie':
        m = len(history) + 1
        k33r = (history[-1][self_index] == 'D') if len(history) > 0 else 0
        j = (history[-1][opp_index] == 'D') if len(history) > 0 else 0.5
        if m > 2:
            index = 2 * ((history[-2][self_index] == 'D') if len(history) > 1 else 1) + ((history[-1][self_index] == 'D') if len(history) > 0 else 1) + 1
            idx = index - 1
            internal_memory[23][0][idx] += float(1 - j)
            internal_memory[23][1][idx] += 1.0
            if internal_memory[23][1][idx] > 0:
                internal_memory[23][2][idx] = internal_memory[23][0][idx] / internal_memory[23][1][idx]
        index = 2 * ((history[-2][self_index] == 'D') if len(history) > 1 else 1) + ((history[-1][self_index] == 'D') if len(history) > 0 else 1) + 1
        idx = index - 1
        if m > 1 and j != ((history[-1][self_index] == 'D') if len(history) > 0 else 1):
            internal_memory[23][3] = False
        if m <= 22:
            ipol = 4
        elif internal_memory[23][3]:
            ipol = 1
        else:
            best = -1.0
            ipol = 1
            for ii in range(6):
                suma = internal_memory[23][4][ii]
                for jj in range(4):
                    suma += internal_memory[23][5][ii][jj] * internal_memory[23][2][jj]
                if suma > best:
                    best = suma
                    ipol = ii + 1
        decision_map = {
            1: [0, 0, 0, 0],
            2: [1, 0, 0, 0],
            3: [1, 0, 1, 0],
            4: [1, 1, 0, 0],
            5: [1, 1, 1, 0],
            6: [1, 1, 1, 1],
        }
        if internal_memory[23][3] and m > 22:
            k33r = 0
        else:
            k33r = decision_map[ipol][idx]
        return 'CD'[k33r]
    elif strategy == 'Second By Hall':
        k71r = (history[-1][self_index] == 'D') if len(history) > 0 else 0
        if (len(history) + 1) == 1:
            return 'C'
        if (len(history) + 1) == 2:
            k71r = 1
            if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 1:
                k71r = 0
            return 'CD'[k71r]
        if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 0:
            internal_memory[172]["ia"] += 1
            if internal_memory[172]["ia"] == 2:
                internal_memory[172]["ia"] = 0
                k71r = 1
            else:
                k71r = 0
        else:
            internal_memory[172]["ib"] += 1
            if internal_memory[172]["ib"] == 2:
                internal_memory[172]["ib"] = 0
                k71r = 1
            else:
                k71r = 0
        return 'CD'[k71r]
    elif strategy == 'Second By Friedland':
        if (len(history) + 1) == 1:
            return 'C'
        if internal_memory[389]["jr"] == 1:
            internal_memory[389]["k74r_val"] = 1
            return 'D'
        if (len(history) + 1) > 2:
            if internal_memory[389]["iold"] == 0:
                if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 0:
                    internal_memory[389]["qca"] += 1.0
                internal_memory[389]["qna"] += 1.0
                internal_memory[389]["alpha"] = internal_memory[389]["qca"] / internal_memory[389]["qna"]
                internal_memory[389]["qca"] *= 0.8
                internal_memory[389]["qna"] *= 0.8
            else:
                if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 0:
                    internal_memory[389]["qcb"] += 1.0
                internal_memory[389]["qnb"] += 1.0
                internal_memory[389]["beta"] = internal_memory[389]["qcb"] / internal_memory[389]["qnb"]
                internal_memory[389]["qcb"] *= 0.8
                internal_memory[389]["qnb"] *= 0.8
        internal_memory[389]["iold"] = internal_memory[389]["k74r_val"]
        if (len(history) + 1) == 37:
            if not (
                internal_memory[389]["js4"] == 0
                or internal_memory[389]["js11"] == 1
                or internal_memory[389]["jt"] <= 10
                or internal_memory[389]["jt"] >= 26
                or internal_memory[389]["jsw"] >= 26
            ):
                internal_memory[389]["jr"] = 1
                internal_memory[389]["k74r_val"] = 1
                return 'D'
        elif (len(history) + 1) > 1:
            if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == internal_memory[389]["jl"]:
                internal_memory[389]["jsm"] += 1
                if internal_memory[389]["jsm"] >= 3:
                    internal_memory[389]["js4"] = 1
                if internal_memory[389]["jsm"] >= 11:
                    internal_memory[389]["js11"] = 1
            else:
                internal_memory[389]["jsw"] += 1
                internal_memory[389]["jsm"] = 1
            internal_memory[389]["jt"] += ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
        internal_memory[389]["jl"] = ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
        polc = 6.0 * internal_memory[389]["alpha"] - 8.0 * internal_memory[389]["beta"] - 2.0
        polalt = 4.0 * internal_memory[389]["alpha"] - 5.0 * internal_memory[389]["beta"] - 1.0
        if polc == 0.0:
            k74r = 0 if polc >= polalt else 1
        elif polalt >= 0.0:
            k74r = 1 - internal_memory[389]["k74r_val"]
        else:
            k74r = 1
        internal_memory[389]["k74r_val"] = k74r
        return 'CD'[k74r]
    elif strategy == 'Second By Hotz':
        if 1 <= (len(history) + 1) < 100:
            probc = 0.1
        elif 100 <= (len(history) + 1) < 200:
            probc = 0.05
        elif 200 <= (len(history) + 1) < 300:
            probc = 0.15
        else:
            probc = 0.0
        return rnd_prob(probc)
    elif strategy == 'Go By Majority / Second By Grisell / Soft Go By Majority':
        if not len(history):
            return 'C'
        return choice_majority([history[i][opp_index] for i in range(len(history))], 'C')
    elif strategy == 'Remorseful Prober':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'D':
            if internal_memory[176]:
                internal_memory[176] = False
                return 'C'
            return 'D'
        if rnd.random() <= 0.9:
            internal_memory[176] = False
            return 'C'
        internal_memory[176] = True
        return 'D'
    elif strategy == 'Loyal':
        if (len(history)) < 2:
            return 'C'
        else:
            return history[-1][opp_index]
    elif strategy == 'Vandal':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'D':
            internal_memory[191] += 1
        if internal_memory[191] >= 3:
            return 'D'
        if (len(history)) % 4 == 3:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Gold Digger':
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[192] += 1
        if (len(history)) < 2:
            return 'D'
        if internal_memory[192] == 0:
            return 'D'
        if (len(history) + 1) == 3:
            return 'C'
        return history[-1][opp_index]
    elif strategy == 'Trusty':
        if (len(history) + 1) > 5:
            if internal_memory[193] == 1:
                return 'D'
            elif history[-1][opp_index] == 'D':
                internal_memory[193] = 1
                return 'D'
            else:
                return 'C'
        else:
            if len(history) > 0:
                if history[-1][opp_index] == 'D':
                    internal_memory[193] = 1
            return 'C'
    elif strategy == 'Mirror':
        if (len(history)) < 2:
            return 'C'
        return history[-2][opp_index]
    elif strategy == 'Bayes Foomii Version':
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[194] += 1
        if (len(history)) < 12:
            return 'C'
        p_d = internal_memory[194] / (len(history))
        if p_d >= 0.55:
            return 'D'
        else:
            return history[-1][opp_index]
    elif strategy == 'Calculator The Nerd Of AI Version':
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[200] += 1
        p_d = (internal_memory[200] / (len(history))) if len(history) > 0 else 0.5
        if p_d > 0.5:
            return 'D'
        return 'C'
    elif strategy == 'Opportunist':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'C':
            return 'D'
        return 'D'
    elif strategy == 'Long Term Strategist':
        if not len(history):
            return 'C'
        inp = tuple([(history[-i][opp_index] == 'D') for i in range(min(3, len(history)), 0, -1)])
        if sum(inp) >= 2:
            return 'D'
        return 'C'
    elif strategy == 'Manipulator':
        if (len(history)) < 3:
            return 'C'
        return 'D'
    elif strategy == 'Retaliate':
        retaliation_threshold = 0.1
        if len(history):
            last_round = (history[-1][self_index], history[-1][opp_index])
            internal_memory[203][last_round] += 1
        CD_count = internal_memory[203][('C', 'D')]
        DC_count = internal_memory[203][('D', 'C')]
        if CD_count > DC_count * retaliation_threshold:
            return 'D'
        return 'C'
    elif strategy == 'Limited Retaliate':
        retaliation_threshold = 0.1
        """
        If the opponent has played D to my C more often than x% of the time
        that I've done the same to him, retaliate by playing D but stop doing
        so once I've hit the retaliation limit.
        """
        if len(history):
            last_round = (history[-1][self_index], history[-1][opp_index])
            internal_memory[204][3][last_round] += 1
        CD_count = internal_memory[204][3][('C', 'D')]
        DC_count = internal_memory[204][3][('D', 'C')]
        if CD_count > DC_count * retaliation_threshold:
            internal_memory[204][0] = True
        else:
            internal_memory[204][0] = False
            internal_memory[204][1] = 0
        if internal_memory[204][0]:
            if internal_memory[204][1] < internal_memory[204][2]:
                internal_memory[204][1] += 1
                return 'D'
            else:
                internal_memory[204][1] = 0
                internal_memory[204][0] = False
        return 'C'
    elif strategy == 'Cooperator Hunter':
        if len(history) > 0:
            if history[-1][opp_index] == 'C':
                internal_memory[205] += 1
        if (len(history) + 1) > 4 and (len(history)) == internal_memory[205]:
            return 'D'
        return 'C'
    elif strategy == 'Defector Hunter':
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[206] += 1
        if (len(history) + 1) > 4 and (len(history)) == internal_memory[206]:
            return 'D'
        return 'C'
    elif strategy == 'Alternator Hunter':
        if (len(history)) < 6:
            return 'C'
        inp = tuple([history[-i][opp_index] for i in range(6, 0, -1)]) 
        if inp not in (("C", "D", "C", "D", "C", "D"), ("D", "C", "D", "C", "D", "C")):
            internal_memory[157] = False
        if internal_memory[157]:
            return 'D'
        return 'C'
    elif strategy == 'Cycler CCCCCD':
        cyclic_pattern = 'CCCCCD'
        return cyclic_pattern[len(history) % len(cyclic_pattern)]
    elif strategy == 'Cycler CCCD':
        cyclic_pattern = 'CCCD'
        return cyclic_pattern[len(history) % len(cyclic_pattern)]
    elif strategy == 'Cycler CCCDCD':
        cyclic_pattern = 'CCCDCD'
        return cyclic_pattern[len(history) % len(cyclic_pattern)]
    elif strategy == 'Cycler DC':
        cyclic_pattern = 'DC'
        return cyclic_pattern[len(history) % len(cyclic_pattern)]
    elif strategy == 'Cycler DDC':
        cyclic_pattern = 'DDC'
        return cyclic_pattern[len(history) % len(cyclic_pattern)]
    elif strategy == 'Worse & Worse / Worse & Worse 1':
        return 'D' if rnd.random() <= (len(history) / 1000) else 'C'
    elif strategy == 'Better & Better':
        return 'C' if rnd.random() <= (len(history) / 1000) else 'D'
    elif strategy == 'Handshake':
        if (len(history)) < 2:
            return ('C', 'D')[len(history)]
        if (len(history) + 1) == 3:
            internal_memory[207] = [history[-i][opp_index] for i in range(2, 0, -1)] 
        if internal_memory[207] == ['C', 'D']:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Fortress 3':
        if not len(history):
            return 'D'
        instruction_string = '1:D1|D2,2:D1|C3,3:C3|D1'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                mode_number, action = item.split(':')
                action_0_str, action_1_str = action.split('|')
                action_0 = [action_0_str[0], action_0_str[1:]]
                action_1 = [action_1_str[0], action_1_str[1:]]
                rule[mode_number] = [action_0, action_1]
            return rule
        instruction = translating(instruction_string)
        fortress_rule = instruction[internal_memory[208]][(history[-1][opp_index] == 'D')]
        internal_memory[208] = fortress_rule[1]
        return fortress_rule[0]
    elif strategy == 'Fortress 4':
        if not len(history):
            return 'D'
        instruction_string = '1:D1|D2,2:D1|D3,3:D1|C4,4:C4|D1'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                mode_number, action = item.split(':')
                action_0_str, action_1_str = action.split('|')
                action_0 = [action_0_str[0], action_0_str[1:]]
                action_1 = [action_1_str[0], action_1_str[1:]]
                rule[mode_number] = [action_0, action_1]
            return rule
        instruction = translating(instruction_string)
        fortress_rule = instruction[internal_memory[209]][(history[-1][opp_index] == 'D')]
        internal_memory[209] = fortress_rule[1]
        return fortress_rule[0]
    elif strategy == 'Tricky Defector':
        if len(history) > 0:
            if history[-1][opp_index] == 'C':
                internal_memory[210] = 1
        if (len(history) + 1) > 3:
            inp = tuple([history[-i][opp_index] for i in range(1, 3 + 1)]) 
            if internal_memory[210] == 1 and 'C' not in inp:
                return 'C'
        return 'D'
    elif strategy == 'Punisher':
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[211][3] += 1
        """
        Begins by playing C, then plays D for an amount of rounds proportional
        to the opponents historical '%' of playing D if the opponent ever
        plays D
        """
        if internal_memory[211][2] >= internal_memory[211][0]:
            internal_memory[211][2] = 0
            internal_memory[211][1] = False
        if internal_memory[211][1]:
            internal_memory[211][2] += 1
            return 'D'
        elif ((history[-1][opp_index] == 'D') if len(history) > 0 else False):
            internal_memory[211][0] = (internal_memory[211][3] * 20) // (len(history))
            internal_memory[211][1] = True
            return 'D'
        return 'C'
    elif strategy == 'Resurrection':
        if not len(history):
            return 'C'
        if len(history) >= 5:
            inp = [history[-i][self_index] for i in range(1, 5 + 1)] 
            if inp == ['D', 'D', 'D', 'D', 'D']:
                return 'D'
        return history[-1][opp_index]
    elif strategy == 'Detective':
        if (len(history)) < 4:
            return ('C', 'D', 'C', 'C')[len(history)]
        elif (len(history) + 1) == 5:
            inp = tuple([(history[-i][opp_index] == 'D') for i in range(4, 0, -1)]) 
            total_D = sum(inp)
            if total_D > 0:
                internal_memory[213] = 1
            else:
                internal_memory[213] = 0
        if internal_memory[213] == 1:
            return history[-1][opp_index]
        else:
            return 'D'
    elif strategy == 'AON2':
        if (len(history)) < 2:
            return 'C'
        return 'C' if rnd.random() <= {('CC', 'CC'): 1, ('CC', 'CD'): 0, ('CC', 'DC'): 0, ('CC', 'DD'): 0, ('CD', 'CC'): 0, ('CD', 'CD'): 1, ('CD', 'DC'): 0, ('CD', 'DD'): 0, ('DC', 'CC'): 0, ('DC', 'CD'): 0, ('DC', 'DC'): 1, ('DC', 'DD'): 0, ('DD', 'CC'): 0, ('DD', 'CD'): 0, ('DD', 'DC'): 0, ('DD', 'DD'): 1}[(history[-2][self_index] + history[-1][self_index], history[-2][opp_index] + history[-1][opp_index])] else 'D'
    elif strategy == 'Delayed AON1':
        if (len(history)) < 2:
            return 'C'
        return 'C' if rnd.random() <= {('CC', 'CC'): 1, ('CC', 'CD'): 0, ('CC', 'DC'): 0, ('CC', 'DD'): 0, ('CD', 'CC'): 0, ('CD', 'CD'): 1, ('CD', 'DC'): 0, ('CD', 'DD'): 1, ('DC', 'CC'): 0, ('DC', 'CD'): 0, ('DC', 'DC'): 1, ('DC', 'DD'): 0, ('DD', 'CC'): 0, ('DD', 'CD'): 1, ('DD', 'DC'): 0, ('DD', 'DD'): 1}[(history[-2][self_index] + history[-1][self_index], history[-2][opp_index] + history[-1][opp_index])] else 'D'
    elif strategy == 'Meta Minority':
        meta_mino_team = [
            call_strategy('First By Tit For Tat', history, opp_index, opp_name, None, None, None, None, None, None),
            call_strategy('Always Cooperate / Cooperator / Dove / Saint', history, opp_index, opp_name, None, None, None, None, None, None),
            call_strategy('Always Defect / Defector / Hawk / Shark', history, opp_index, opp_name, None, None, None, None, None, None),
            call_strategy('Random', history, opp_index, opp_name, None, None, None, None, None, None),
            call_strategy('First By Grudger / Grim Trigger', history, opp_index, opp_name, internal_memory[425], None, None, None, None, None),
            call_strategy('Pavlov / Win-Stay, Lose-Shift', history, opp_index, opp_name, None, None, None, None, None, None),
            call_strategy('Generous Tit For Tat', history, opp_index, opp_name, None, None, None, None, None, None),
            call_strategy('Sample / Second By Tit For 2 Tats', history, opp_index, opp_name, None, None, None, None, None, None),
        ]
        meta_mino_majority = sum((1 if i == 'C' else -1) for i in meta_mino_team)
        if meta_mino_majority < 0:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Meta Mixer':
        meta_mixe_team = [
            call_strategy('First By Tit For Tat', history, opp_index, opp_name, None, None, None, None, None, None),
            call_strategy('Always Cooperate / Cooperator / Dove / Saint', history, opp_index, opp_name, None, None, None, None, None, None),
            call_strategy('Always Defect / Defector / Hawk / Shark', history, opp_index, opp_name, None, None, None, None, None, None),
            call_strategy('Random', history, opp_index, opp_name, None, None, None, None, None, None),
            call_strategy('First By Grudger / Grim Trigger', history, opp_index, opp_name, internal_memory[426], None, None, None, None, None),
            call_strategy('Pavlov / Win-Stay, Lose-Shift', history, opp_index, opp_name, None, None, None, None, None, None),
            call_strategy('Generous Tit For Tat', history, opp_index, opp_name, None, None, None, None, None, None),
            call_strategy('Sample / Second By Tit For 2 Tats', history, opp_index, opp_name, None, None, None, None, None, None),
        ]
        return rnd.choice(meta_mixe_team)
    elif strategy == 'Win-Shift, Lose-Stay':
        if not len(history):
            return 'D'
        if history[-1][opp_index] != history[-1][self_index]:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Adaptive':
        if len(history) > 0:
            internal_memory[231][0][history[-1][opp_index]] += score[self_index] - internal_memory[231][1]
        internal_memory[231][1] = score[self_index]
        if (len(history)) < 11:
            return 'C' if (len(history)) < 6 else 'D'
        if internal_memory[231][0]['C'] > internal_memory[231][0]['D']:
            return 'C'
        return 'D'
    elif strategy == 'Adaptor Brief':
        perr = 0.01
        delta = {('C', 'C'):0.0, ('C', 'D'):0.992107, ('D', 'C'):-1.001505, ('D', 'D'):-0.638734}
        if len(history) > 0:
            last_round = (history[-1][self_index], history[-1][opp_index])
            internal_memory[233] += delta[last_round]
        def step(x):
            return (1 if x > 0 else 0) if x != 0 else 0.5
        p = perr + (1 - 2 * perr) * (
            step(internal_memory[233] + 1) - step(internal_memory[233] - 1)
        )
        if rnd.random() <= p:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Adaptor Long':
        perr = 0.01
        delta = {('C', 'C'): 0.0, ('C', 'D'): 1.858883, ('D', 'C'): 1.888159, ('D', 'D'): -0.995703}
        if len(history) > 0:
            last_round = (history[-1][self_index], history[-1][opp_index])
            internal_memory[234] += delta[last_round]
        def step(x):
            return (1 if x > 0 else 0) if x != 0 else 0.5
        p = perr + (1 - 2 * perr) * (
            step(internal_memory[234] + 1) - step(internal_memory[234] - 1)
        )
        if rnd.random() <= p:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Adaptive Pavlov 2006':
        if (len(history)) < 6:
            if not len(history):
                return 'C'
            return history[-1][opp_index]
        inp = [history[-i][opp_index] for i in range(1, 6 + 1)] 
        if (len(history)) % 6 == 0:
            if inp == ['C'] * 6:
                internal_memory[235] = "Cooperative"
            if inp == ['D'] * 6:
                internal_memory[235] = "ALLD"
            if inp == ['D', 'C', 'D', 'C', 'D', 'C']:
                internal_memory[235] = "STFT"
            if inp == ['D', 'D', 'C', 'D', 'D', 'C']:
                internal_memory[235] = "PavlovD"
            if not internal_memory[235]:
                internal_memory[235] = "Random"
        if internal_memory[235] in ["Random", "ALLD"]:
            return 'D'
        if internal_memory[235] == "STFT":
            if (len(history)) % 6 in [0, 1]:
                return 'C'
            if history[-1][opp_index] == 'D':
                return 'D'
        if internal_memory[235] == "PavlovD":
            if (len(history)) % 6 == 0:
                return 'D'
        if internal_memory[235] == "Cooperative":
            if history[-1][opp_index] == 'D':
                return 'D'
        return 'C'
    elif strategy == 'Adaptive Pavlov 2011':
        if (len(history)) < 6:
            if not len(history):
                return 'C'
            return history[-1][opp_index]
        inp = [history[-i][opp_index] for i in range(1, 6 + 1)] 
        if (len(history)) % 6 == 0:
            if inp == ['C'] * 6:
                internal_memory[236] = "Cooperative"
            if inp.count('D') >= 4:
                internal_memory[236] = "ALLD"
            if inp.count('D') == 3:
                internal_memory[236] = "STFT"
            if not internal_memory[236]:
                internal_memory[236] = "Random"
        if internal_memory[236] in ["Random", "ALLD"]:
            return 'D'
        if internal_memory[236] == "STFT":
            return 'D' if (history[-2][opp_index], history[-1][opp_index]) == ('D', 'D') else 'C'
        if internal_memory[236] == "Cooperative":
            return history[-1][opp_index]
    elif strategy == 'Appeaser':
        if not len(history): return 'C'
        if history[-1][opp_index] == 'D':
            return flip(history[-1][self_index])
        return history[-1][self_index]
    elif strategy == 'Average Copier':
        if not len(history):
            return rnd_prob(0.5)
        if history[-1][opp_index] == 'C':
            internal_memory[237] += 1
        p_c = internal_memory[237] / (len(history))
        return 'C' if rnd.random() <= p_c else 'D'
    elif strategy == 'Second By Grofman':
        if (len(history)) < 2:
            return 'C'
        elif 2 < (len(history)) < 7:
            return history[-1][opp_index]
        else:
            t_d = sum([(history[-i][opp_index] == 'D') for i in range(2, min(len(history), 8) + 1)]) 
            if history[-1][opp_index] == 'C' and t_d <= 2:
                return 'C'
            if history[-1][opp_index] == 'D' and t_d <= 1:
                return 'C'
            return 'D'
    elif strategy == 'Second By Tideman & Chieruzzi':
        def _fresh_start():
            """Give the opponent a fresh start by forgetting the past"""
            score[self_index] = 0
            score[opp_index] = 0
            internal_memory[238][2] = 0
            internal_memory[238][3] = 0
        """Actual strategy definition that determines player's action."""
        current_round = (len(history) + 1)
        if current_round == 1:
            return 'C'
        if history[-1][opp_index] == 'D':
            internal_memory[238][4] += 1
        if internal_memory[238][1]:
            _fresh_start()
            internal_memory[238][0] = current_round
            internal_memory[238][1] = False
            return 'C'  
        opponent_CDd = False
        opponent_two_turns_ago = 'C'  
        if (len(history)) >= 2:
            opponent_two_turns_ago = history[-2][opp_index]
        if opponent_two_turns_ago == 'C' and history[-1][opp_index] == 'D':
            opponent_CDd = True
            internal_memory[238][2] += internal_memory[238][3]
            internal_memory[238][3] += 5
        if score[self_index] - score[opp_index] >= internal_memory[238][2]:
            return 'C'
        if (not opponent_CDd) and current_round - internal_memory[238][0] >= 10:
            N = len(history)
            std_deviation = (N ** (1 / 2)) / 2
            lower = N / 2 - 3 * std_deviation
            upper = N / 2 + 3 * std_deviation
            if internal_memory[238][4] <= lower or internal_memory[238][4] >= upper:
                internal_memory[238][1] = True
                return 'C'  
        return 'D'
    elif strategy == 'Back Stabber':
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[239] += 1
        if internal_memory[239] > 3:
            return 'D'
        return 'C'
    elif strategy == 'Double Crosser':
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[240][0] += 1
                if (len(history)) < 7:
                    internal_memory[240][1] = 1
        if (internal_memory[240][1] == 0) and (7 < (len(history) + 1) < 180):
            return 'D' if (history[-2][opp_index], history[-1][opp_index]) == ('D', 'D') else 'C'
        if internal_memory[240][0] > 3:
            return 'D'
        return 'C'
    elif strategy == 'Bush Mosteller':
        def stimulus_update():
            """
            Updates the stimulus attribute based on the opponent's history. Used by
            the strategy.
            Parameters
            opponent : axelrod.Player
                The current opponent
            """
            last_round = (history[-1][self_index], history[-1][opp_index])
            previous_play = score[self_index] - internal_memory[241][5]
            internal_memory[241][3] = (previous_play - internal_memory[241][2]) / abs(
                (max(5,3,1,0) - internal_memory[241][2])
            )
            if internal_memory[241][3] < -1:
                internal_memory[241][3] = -1
            if history[-1][self_index] == 'C':
                if internal_memory[241][3] >= 0:
                    internal_memory[241][0] += (
                        internal_memory[241][4] * internal_memory[241][3] * (1 - internal_memory[241][0])
                    )
                elif internal_memory[241][3] < 0:
                    internal_memory[241][0] += (
                        internal_memory[241][4] * internal_memory[241][3] * internal_memory[241][0]
                    )
            if history[-1][self_index] == 'D':
                if internal_memory[241][3] >= 0:
                    internal_memory[241][1] += (
                        internal_memory[241][4] * internal_memory[241][3] * (1 - internal_memory[241][1])
                    )
                elif internal_memory[241][3] < 0:
                    internal_memory[241][1] += (
                        internal_memory[241][4] * internal_memory[241][3] * internal_memory[241][1]
                    )
        """Actual strategy definition that determines player's action."""
        if (len(history)) == 0:
            return 'C' if rnd.random() <= (
                internal_memory[241][0] / (internal_memory[241][0] + internal_memory[241][1])
            ) else 'D'
        stimulus_update()
        return 'C' if rnd.random() <= (
            internal_memory[241][0] / (internal_memory[241][0] + internal_memory[241][1])
        ) else 'D'
    elif strategy == 'Calculator':
        if (len(history)) < 21:
            if len(history) > 0:
                internal_memory[242][0][(len(history) + 1) - 2] = history[-1][opp_index]
            if (len(history) + 1) < 21:
                if not len(history):
                    return 'C'
                return history[-1][opp_index] if rnd.random() > 0.1 else 'D'
        if (len(history) + 1) == 21:
            for i in range(10):
                Calculator_n = (i + 1)
                is_cycle = True
                Calculator_list = internal_memory[242][0][:(Calculator_n + 1)]
                Calculator_index = len(internal_memory[242][0]) % Calculator_n
                for j in range(len(internal_memory[242][0])):
                    if internal_memory[242][0][j] != Calculator_list[j % Calculator_n]:
                        is_cycle = False
                if is_cycle:
                    internal_memory[242][1] = is_cycle
                    break
        if internal_memory[242][1]:
            return 'D'
        return history[-1][opp_index]
    elif strategy == 'Anti Cycler':
        if (len(history)) < len(internal_memory[243][2]):
            return internal_memory[243][2].pop(0)
        if internal_memory[243][1] < internal_memory[243][0]:
            internal_memory[243][1] += 1
            return 'C'
        else:
            internal_memory[243][0] += 1
            internal_memory[243][1] = 0
            return 'D'
    elif strategy == 'DBS / Derived Belief Strategy':
        class Node(object):
            def get_siblings(self): raise NotImplementedError()
            def is_stochastic(self): raise NotImplementedError()
        class StochasticNode(Node):
            def __init__(self, own_action, pC, depth):
                self.pC = pC
                self.depth = depth
                self.own_action = own_action
            def get_siblings(self):
                return DeterministicNode(self.own_action, 'C', self.depth + 1), \
                       DeterministicNode(self.own_action, 'D', self.depth + 1)
            def is_stochastic(self):
                return True
        class DeterministicNode(Node):
            def __init__(self, action1, action2, depth):
                self.action1 = action1
                self.action2 = action2
                self.depth = depth
            def get_siblings(self, policy):
                return StochasticNode('C', policy[(self.action1, self.action2)], self.depth), \
                       StochasticNode('D', policy[(self.action1, self.action2)], self.depth)
            def is_stochastic(self):
                return False
            def get_value(self):
                values = {('C', 'C'): self_payoff_matrix['CC'], ('C', 'D'): self_payoff_matrix['CD'], ('D', 'C'): self_payoff_matrix['DC'], ('D', 'D'): self_payoff_matrix['DD']}
                return values[(self.action1, self.action2)]
        def action_to_int(action):
            return 1 if action == 'C' else 0
        def minimax_tree_search(begin_node, policy, max_depth):
            if begin_node.is_stochastic():
                siblings = begin_node.get_siblings()
                return begin_node.pC * minimax_tree_search(siblings[0], policy, max_depth) + \
                       (1 - begin_node.pC) * minimax_tree_search(siblings[1], policy, max_depth)
            else:
                if begin_node.depth == max_depth:
                    return begin_node.get_value()
                elif begin_node.depth == 0:
                    siblings = begin_node.get_siblings(policy)
                    return (
                        minimax_tree_search(siblings[0], policy, max_depth) + begin_node.get_value(),
                        minimax_tree_search(siblings[1], policy, max_depth) + begin_node.get_value(),
                    )
                elif begin_node.depth < max_depth:
                    siblings = begin_node.get_siblings(policy)
                    a = minimax_tree_search(siblings[0], policy, max_depth)
                    b = minimax_tree_search(siblings[1], policy, max_depth)
                    return max(a, b) + begin_node.get_value()
        def move_gen(outcome, policy, depth_search_tree=5):
            current_node = DeterministicNode(outcome[0], outcome[1], depth=0)
            values_of_choices = minimax_tree_search(current_node, policy, depth_search_tree)
            actions_tuple = ('C', 'D')
            return actions_tuple[values_of_choices.index(max(values_of_choices))]
        dbs = internal_memory[245]
        my_idx = self_index
        if len(history) >= 2:
            two_moves_ago = (history[-2][my_idx], history[-2][opp_index])
            op_last_move = history[-1][opp_index]
            for outcome, GF in dbs['history_by_cond'].items():
                G, F = GF
                if outcome == two_moves_ago:
                    G.append(1 if op_last_move == 'C' else 0)
                    F.append(1)
                else:
                    G.append(0)
                    F.append(0)
            r_plus = (two_moves_ago, op_last_move)
            r_minus = (two_moves_ago, 'D' if op_last_move == 'C' else 'C')
            if r_plus[0] not in dbs['Rc'].keys():
                opposite_action = 0 if r_plus[1] == 'C' else 1
                k = 1
                count = 0
                cond_G = dbs['history_by_cond'][r_plus[0]][0]
                cond_F = dbs['history_by_cond'][r_plus[0]][1]
                while k < len(cond_G) and not (cond_G[1:][-k] == opposite_action and cond_F[1:][-k] == 1):
                    if cond_F[1:][-k] == 1:
                        count += 1
                    k += 1
                if count >= dbs['promotion_threshold']:
                    dbs['Rc'][r_plus[0]] = action_to_int(r_plus[1])
                    dbs['violation_counts'][r_plus[0]] = 0
            if r_plus[0] in dbs['Rc'].keys():
                to_check = 'C' if dbs['Rc'][r_plus[0]] == 1 else 'D'
                if r_plus[1] == to_check:
                    dbs['violation_counts'][r_plus[0]] = 0
                elif r_minus[1] == to_check:
                    dbs['violation_counts'][r_plus[0]] = dbs['violation_counts'].get(r_plus[0], 0) + 1
                    if dbs['violation_counts'].get(r_minus[0], 0) >= dbs['violation_threshold']:
                        dbs['Rd'].update(dbs['Rc'])
                        dbs['Rc'].clear()
                        dbs['violation_counts'].clear()
                        dbs['v'] = 0
            r_plus_in_Rc = r_plus[0] in dbs['Rc'].keys() and dbs['Rc'][r_plus[0]] == action_to_int(r_plus[1])
            r_minus_in_Rd = r_minus[0] in dbs['Rd'].keys() and dbs['Rd'][r_minus[0]] == action_to_int(r_minus[1])
            if r_minus_in_Rd:
                dbs['v'] += 1
            if (dbs['v'] > dbs['reject_threshold']) or (r_plus_in_Rc and r_minus_in_Rd):
                dbs['Rd'].clear()
                dbs['v'] = 0
            Rp = {}
            all_cond = [('C', 'C'), ('C', 'D'), ('D', 'C'), ('D', 'D')]
            for outcome in all_cond:
                if (outcome not in dbs['Rc'].keys()) and (outcome not in dbs['Rd'].keys()):
                    G = dbs['history_by_cond'][outcome][0]
                    F = dbs['history_by_cond'][outcome][1]
                    discounted_g = 0
                    discounted_f = 0
                    alpha_k = 1
                    for g, f in zip(G[::-1], F[::-1]):
                        discounted_g += alpha_k * g
                        discounted_f += alpha_k * f
                        alpha_k = dbs['alpha'] * alpha_k
                    Rp[outcome] = discounted_g / discounted_f if discounted_f != 0 else 0.5
            dbs['Pi'] = {}
            dbs['Pi'].update(dbs['Rc'])
            dbs['Pi'].update(dbs['Rd'])
            dbs['Pi'].update(Rp)
        last_outcome = (history[-1][my_idx], history[-1][opp_index]) if len(history) > 0 else ('C', 'C')
        return move_gen(last_outcome, dbs['Pi'], depth_search_tree=dbs['tree_depth'])
    elif strategy == 'Doubler':
        if len(history) > 0:
            if history[-1][opp_index] == 'C':
                internal_memory[246][0] += 1
            else:
                internal_memory[246][1] += 1
                if internal_memory[246][0] <= (internal_memory[246][1] * 2):
                    return 'D'
        return 'C'
    elif strategy == 'Predator':
        if not len(history):
            return 'C'
        instruction_string = '0:D0|D1,1:D2|D3,2:C4|D3,3:D5|C4,4:C2|D6,5:D7|D3,6:C7|D7,7:D8|D7,8:D8|D6'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                mode_number, action = item.split(':')
                action_0_str, action_1_str = action.split('|')
                action_0 = [action_0_str[0], action_0_str[1:]]
                action_1 = [action_1_str[0], action_1_str[1:]]
                rule[mode_number] = [action_0, action_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[internal_memory[247]][(history[-1][opp_index] == 'D')]
        internal_memory[247] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'Pun 1':
        if not len(history):
            return 'D'
        instruction_string = '1:C2|C2,2:C1|D1'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                mode_number, action = item.split(':')
                action_0_str, action_1_str = action.split('|')
                action_0 = [action_0_str[0], action_0_str[1:]]
                action_1 = [action_1_str[0], action_1_str[1:]]
                rule[mode_number] = [action_0, action_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[internal_memory[248]][(history[-1][opp_index] == 'D')]
        internal_memory[248] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'Raider':
        if not len(history):
            return 'D'
        instruction_string = '0:D2|D2,1:C1|D1,2:D0|C3,3:D0|C1'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                mode_number, action = item.split(':')
                action_0_str, action_1_str = action.split('|')
                action_0 = [action_0_str[0], action_0_str[1:]]
                action_1 = [action_1_str[0], action_1_str[1:]]
                rule[mode_number] = [action_0, action_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[internal_memory[249]][(history[-1][opp_index] == 'D')]
        internal_memory[249] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'Ripoff':
        if not len(history):
            return 'D'
        instruction_string = '1:C2|C3,2:D1|C3,3:C3|D3'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                mode_number, action = item.split(':')
                action_0_str, action_1_str = action.split('|')
                action_0 = [action_0_str[0], action_0_str[1:]]
                action_1 = [action_1_str[0], action_1_str[1:]]
                rule[mode_number] = [action_0, action_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[internal_memory[250]][(history[-1][opp_index] == 'D')]
        internal_memory[250] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'Usually Cooperates':
        if not len(history):
            return 'C'
        instruction_string = '1:C1|C2,2:D1|C1'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                mode_number, action = item.split(':')
                action_0_str, action_1_str = action.split('|')
                action_0 = [action_0_str[0], action_0_str[1:]]
                action_1 = [action_1_str[0], action_1_str[1:]]
                rule[mode_number] = [action_0, action_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[internal_memory[251]][(history[-1][opp_index] == 'D')]
        internal_memory[251] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'Usually Defects':
        if not len(history):
            return 'D'
        instruction_string = '1:D2|D1,2:D1|C1'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                mode_number, action = item.split(':')
                action_0_str, action_1_str = action.split('|')
                action_0 = [action_0_str[0], action_0_str[1:]]
                action_1 = [action_1_str[0], action_1_str[1:]]
                rule[mode_number] = [action_0, action_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[internal_memory[252]][(history[-1][opp_index] == 'D')]
        internal_memory[252] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'Solution B1':
        if not len(history):
            return 'D'
        instruction_string = '1:D2|D1,2:C2|C3,3:C3|C3'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                mode_number, action = item.split(':')
                action_0_str, action_1_str = action.split('|')
                action_0 = [action_0_str[0], action_0_str[1:]]
                action_1 = [action_1_str[0], action_1_str[1:]]
                rule[mode_number] = [action_0, action_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[internal_memory[253]][(history[-1][opp_index] == 'D')]
        internal_memory[253] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'Solution B5':
        if not len(history):
            return 'D'
        instruction_string = '1:C2|D6,2:C2|D3,3:C6|D1,4:C3|D6,5:D5|D4,6:C3|D5'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                mode_number, action = item.split(':')
                action_0_str, action_1_str = action.split('|')
                action_0 = [action_0_str[0], action_0_str[1:]]
                action_1 = [action_1_str[0], action_1_str[1:]]
                rule[mode_number] = [action_0, action_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[internal_memory[254]][(history[-1][opp_index] == 'D')]
        internal_memory[254] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'Thumper':
        if not len(history):
            return 'C'
        instruction_string = '1:C1|D2,2:D1|D1'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                mode_number, action = item.split(':')
                action_0_str, action_1_str = action.split('|')
                action_0 = [action_0_str[0], action_0_str[1:]]
                action_1 = [action_1_str[0], action_1_str[1:]]
                rule[mode_number] = [action_0, action_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[internal_memory[255]][(history[-1][opp_index] == 'D')]
        internal_memory[255] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'Evolved FSM 4':
        if not len(history):
            return 'C'
        instruction_string = '0:C0|D2,1:D3|C0,2:D2|C1,3:D3|D1'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                mode_number, action = item.split(':')
                action_0_str, action_1_str = action.split('|')
                action_0 = [action_0_str[0], action_0_str[1:]]
                action_1 = [action_1_str[0], action_1_str[1:]]
                rule[mode_number] = [action_0, action_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[internal_memory[256]][(history[-1][opp_index] == 'D')]
        internal_memory[256] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'Evolved FSM 16':
        if not len(history):
            return 'C'
        instruction_string = '0:C0|D12,1:D3|C6,2:D2|D14,3:D3|D3,5:D12|D10,6:C5|D12,7:D3|C1,8:C5|C5,10:D11|C8,11:D15|D5,12:C8|D11,13:D13|D7,14:D13|D13,15:D15|C2'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                mode_number, action = item.split(':')
                action_0_str, action_1_str = action.split('|')
                action_0 = [action_0_str[0], action_0_str[1:]]
                action_1 = [action_1_str[0], action_1_str[1:]]
                rule[mode_number] = [action_0, action_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[internal_memory[257]][(history[-1][opp_index] == 'D')]
        internal_memory[257] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'Evolved FSM 16 Noise 05':
        if not len(history):
            return 'C'
        instruction_string = '0:C8|D3,1:C13|D15,2:C12|D3,3:C10|D3,4:D5|D4,5:D4|D10,6:C8|D6,8:C2|D4,10:D4|D1,11:D14|C13,12:C13|C2,13:C13|C6,14:D3|D13,15:D5|C11'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                mode_number, action = item.split(':')
                action_0_str, action_1_str = action.split('|')
                action_0 = [action_0_str[0], action_0_str[1:]]
                action_1 = [action_1_str[0], action_1_str[1:]]
                rule[mode_number] = [action_0, action_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[internal_memory[258]][(history[-1][opp_index] == 'D')]
        internal_memory[258] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'TF 1':
        if not len(history):
            return 'C'
        instruction_string = '0:C7|C1,1:D11|D11,2:D8|C8,3:C3|D12,4:C6|C3,5:C11|D8,6:D13|C14,7:D4|D2,8:D14|D8,9:C0|D10,10:C8|C15,11:D6|D5,12:D6|D9,13:D9|D8,14:D8|D13,15:C4|C5'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                mode_number, action = item.split(':')
                action_0_str, action_1_str = action.split('|')
                action_0 = [action_0_str[0], action_0_str[1:]]
                action_1 = [action_1_str[0], action_1_str[1:]]
                rule[mode_number] = [action_0, action_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[internal_memory[259]][(history[-1][opp_index] == 'D')]
        internal_memory[259] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'TF 2':
        if not len(history):
            return 'C'
        instruction_string = '0:D13|D12,1:D3|D4,2:D14|D9,3:C0|D1,4:D1|D2,7:D12|D2,8:D7|D9,9:D8|D0,10:C2|C15,11:D7|D13,12:C3|D8,13:C7|D10,14:D10|D7,15:C15|D11'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                mode_number, action = item.split(':')
                action_0_str, action_1_str = action.split('|')
                action_0 = [action_0_str[0], action_0_str[1:]]
                action_1 = [action_1_str[0], action_1_str[1:]]
                rule[mode_number] = [action_0, action_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[internal_memory[260]][(history[-1][opp_index] == 'D')]
        internal_memory[260] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'TF 3':
        if not len(history):
            return 'C'
        instruction_string = '0:C0|C3,1:D5|C0,2:C3|D2,3:D4|D6,4:C3|D1,5:C6|D3,6:D6|D6,7:D7|C5'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                mode_number, action = item.split(':')
                action_0_str, action_1_str = action.split('|')
                action_0 = [action_0_str[0], action_0_str[1:]]
                action_1 = [action_1_str[0], action_1_str[1:]]
                rule[mode_number] = [action_0, action_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[internal_memory[261]][(history[-1][opp_index] == 'D')]
        internal_memory[261] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'Evolved FSM 6':
        if not len(history):
            return 'C'
        instruction_string = '2:D2|D7,3:D6|C4,4:C4|D6,5:D2|D7,6:C3|D5,7:C2|D3'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                mode_number, action = item.split(':')
                action_0_str, action_1_str = action.split('|')
                action_0 = [action_0_str[0], action_0_str[1:]]
                action_1 = [action_1_str[0], action_1_str[1:]]
                rule[mode_number] = [action_0, action_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[internal_memory[262]][(history[-1][opp_index] == 'D')]
        internal_memory[262] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'Forgiver':
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[263] += 1
        t_d = internal_memory[263]
        if t_d > (len(history) / 10):
            return 'D'
        return 'C'
    elif strategy == 'Forgiving Tit For Tat':
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[264] += 1
        t_d = internal_memory[264]
        if t_d > (len(history) / 10):
            return history[-1][opp_index]
        return 'C'
    elif strategy == 'Frequency Analyzer':
        """
        Parameters
        ----------
        p, float
            The probability to cooperate
        """
        def update_table():
            if internal_memory[265][2] in internal_memory[265][1].keys():
                results = internal_memory[265][1][internal_memory[265][2]]
                results.append(history[-1][opp_index])
                internal_memory[265][1][internal_memory[265][2]] = results
            else:
                internal_memory[265][1][internal_memory[265][2]] = [history[-1][opp_index]]
        """This is the actual strategy"""
        if (len(history)) > 5:
            internal_memory[265][2] = (
                history[-3][opp_index]
                + history[-3][self_index]
                + history[-2][opp_index]
                + history[-2][self_index]
            )
            internal_memory[265][3] = (
                history[-2][opp_index]
                + history[-2][self_index]
                + history[-1][opp_index]
                + history[-1][self_index]
            )
            update_table()
        if ((len(history)) < 30) or (
            internal_memory[265][3] not in internal_memory[265][1]
        ):
            if not len(history):
                return 'C'
            if history[-1][opp_index] == 'D':
                return 'D'
            return 'C'
        results = internal_memory[265][1][internal_memory[265][3]]
        cooperates = sum((i == 'C') for i in results)
        if (cooperates / (len(history))) > internal_memory[265][0]:
            return 'C'
        return 'D'
    elif strategy == 'PSO Gambler Mem 1':
        A = {('C', 'C'): 1.0, ('C', 'D'): 0.52173487, ('D', 'C'): 0.0, ('D', 'D'): 0.12050939}[(history[-1][self_index], history[-1][opp_index]) if len(history) > 0 else ('C', 'C')]
        return 'C' if rnd.random() <= A else 'D'
    elif strategy == 'PSO Gambler 1_1_1':
        if not len(history):
            return 'C'
        if (len(history) + 1) == 2:
            internal_memory[266] = history[-1][opp_index]
        A = {('C', 'C', 'C'): 1.0, ('C', 'C', 'D'): 1.0, ('C', 'D', 'C'): 0.12304797, ('C', 'D', 'D'): 0.57740178, ('D', 'C', 'C'): 0.0, ('D', 'C', 'D'): 0.0, ('D', 'D', 'C'): 0.13581423, ('D', 'D', 'D'): 0.11886807}[(history[-1][self_index], history[-1][opp_index], internal_memory[266])]
        return 'C' if rnd.random() <= A else 'D'
    elif strategy == 'PSO Gambler 2_2_2':
        if (len(history) + 1) < 3:
            return 'C'
        elif (len(history) + 1) == 3:
            internal_memory[267] = history[-2][opp_index] + history[-1][opp_index]
        A = {('CC', 'CC', 'CC'): 1.0, ('CC', 'CC', 'CD'): 1.0, ('CC', 'CC', 'DC'): 1.0, ('CC', 'CC', 'DD'): 0.0, ('CC', 'CD', 'CC'): 1.0, ('CC', 'CD', 'CD'): 0.95280465, ('CC', 'CD', 'DC'): 0.0, ('CC', 'CD', 'DD'): 0.0, ('CC', 'DC', 'CC'): 0.0, ('CC', 'DC', 'CD'): 0.80897541, ('CC', 'DC', 'DC'): 0.0, ('CC', 'DC', 'DD'): 0.0, ('CC', 'DD', 'CC'): 0.02126434, ('CC', 'DD', 'CD'): 0.0, ('CC', 'DD', 'DC'): 0.43278586, ('CC', 'DD', 'DD'): 0.0, ('CD', 'CC', 'CC'): 0.0, ('CD', 'CC', 'CD'): 0.0, ('CD', 'CC', 'DC'): 1.0, ('CD', 'CC', 'DD'): 0.15140743, ('CD', 'CD', 'CC'): 1.0, ('CD', 'CD', 'CD'): 0.0, ('CD', 'CD', 'DC'): 0.0, ('CD', 'CD', 'DD'): 0.0, ('CD', 'DC', 'CC'): 1.0, ('CD', 'DC', 'CD'): 0.0, ('CD', 'DC', 'DC'): 0.23563137, ('CD', 'DC', 'DD'): 0.0, ('CD', 'DD', 'CC'): 0.0, ('CD', 'DD', 'CD'): 0.65147565, ('CD', 'DD', 'DC'): 1.0, ('CD', 'DD', 'DD'): 0.0, ('DC', 'CC', 'CC'): 0.0, ('DC', 'CC', 'CD'): 0.15412392, ('DC', 'CC', 'DC'): 1.0, ('DC', 'CC', 'DD'): 0.0, ('DC', 'CD', 'CC'): 0.0, ('DC', 'CD', 'CD'): 0.24922166, ('DC', 'CD', 'DC'): 1.0, ('DC', 'CD', 'DD'): 0.0, ('DC', 'DC', 'CC'): 0.0, ('DC', 'DC', 'CD'): 0.0, ('DC', 'DC', 'DC'): 0.00227615, ('DC', 'DC', 'DD'): 0.0, ('DC', 'DD', 'CC'): 0.0, ('DC', 'DD', 'CD'): 0.0, ('DC', 'DD', 'DC'): 0.0, ('DC', 'DD', 'DD'): 1.0, ('DD', 'CC', 'CC'): 0.0, ('DD', 'CC', 'CD'): 0.0, ('DD', 'CC', 'DC'): 0.0, ('DD', 'CC', 'DD'): 0.0, ('DD', 'CD', 'CC'): 0.0, ('DD', 'CD', 'CD'): 0.0, ('DD', 'CD', 'DC'): 0.0, ('DD', 'CD', 'DD'): 1.0, ('DD', 'DC', 'CC'): 0.0, ('DD', 'DC', 'CD'): 0.0, ('DD', 'DC', 'DC'): 0.0, ('DD', 'DC', 'DD'): 0.77344942, ('DD', 'DD', 'CC'): 1.0, ('DD', 'DD', 'CD'): 0.24523149, ('DD', 'DD', 'DC'): 1.0, ('DD', 'DD', 'DD'): 0.0}[(history[-2][self_index] + history[-1][self_index], history[-2][opp_index] + history[-1][opp_index], internal_memory[267])]
        return 'C' if rnd.random() <= A else 'D'
    elif strategy == 'PSO Gambler 2_2_2 Noise 05':
        if (len(history) + 1) < 3:
            return 'C'
        elif (len(history) + 1) == 3:
            internal_memory[268] = history[-2][opp_index] + history[-1][opp_index]
        A = {('CC', 'CC', 'CC'): 1.0, ('CC', 'CC', 'CD'): 1.0, ('CC', 'CC', 'DC'): 0.0, ('CC', 'CC', 'DD'): 0.0, ('CC', 'CD', 'CC'): 0.0, ('CC', 'CD', 'CD'): 1.0, ('CC', 'CD', 'DC'): 0.98603825, ('CC', 'CD', 'DD'): 1.0, ('CC', 'DC', 'CC'): 1.0, ('CC', 'DC', 'CD'): 0.0, ('CC', 'DC', 'DC'): 0.0, ('CC', 'DC', 'DD'): 0.16240799, ('CC', 'DD', 'CC'): 0.63548102, ('CC', 'DD', 'CD'): 0.0, ('CC', 'DD', 'DC'): 0.0, ('CC', 'DD', 'DD'): 0.0, ('CD', 'CC', 'CC'): 1.0, ('CD', 'CC', 'CD'): 0.0, ('CD', 'CC', 'DC'): 1.0, ('CD', 'CC', 'DD'): 0.0, ('CD', 'CD', 'CC'): 1.0, ('CD', 'CD', 'CD'): 0.13863175, ('CD', 'CD', 'DC'): 0.06434619, ('CD', 'CD', 'DD'): 1.0, ('CD', 'DC', 'CC'): 1.0, ('CD', 'DC', 'CD'): 1.0, ('CD', 'DC', 'DC'): 1.0, ('CD', 'DC', 'DD'): 1.0, ('CD', 'DD', 'CC'): 0.0, ('CD', 'DD', 'CD'): 0.7724137, ('CD', 'DD', 'DC'): 1.0, ('CD', 'DD', 'DD'): 0.0, ('DC', 'CC', 'CC'): 0.0, ('DC', 'CC', 'CD'): 0.0, ('DC', 'CC', 'DC'): 1.0, ('DC', 'CC', 'DD'): 0.0, ('DC', 'CD', 'CC'): 1.0, ('DC', 'CD', 'CD'): 1.0, ('DC', 'CD', 'DC'): 0.50999729, ('DC', 'CD', 'DD'): 1.0, ('DC', 'DC', 'CC'): 0.0, ('DC', 'DC', 'CD'): 0.0, ('DC', 'DC', 'DC'): 0.00524508, ('DC', 'DC', 'DD'): 0.87463905, ('DC', 'DD', 'CC'): 0.0, ('DC', 'DD', 'CD'): 0.07127653, ('DC', 'DD', 'DC'): 1.0, ('DC', 'DD', 'DD'): 0.0, ('DD', 'CC', 'CC'): 1.0, ('DD', 'CC', 'CD'): 0.0, ('DD', 'CC', 'DC'): 1.0, ('DD', 'CC', 'DD'): 0.0, ('DD', 'CD', 'CC'): 0.0, ('DD', 'CD', 'CD'): 1.0, ('DD', 'CD', 'DC'): 1.0, ('DD', 'CD', 'DD'): 1.0, ('DD', 'DC', 'CC'): 0.0, ('DD', 'DC', 'CD'): 0.28124022, ('DD', 'DC', 'DC'): 1.0, ('DD', 'DC', 'DD'): 0.0, ('DD', 'DD', 'CC'): 0.0, ('DD', 'DD', 'CD'): 0.0, ('DD', 'DD', 'DC'): 1.0, ('DD', 'DD', 'DD'): 0.0}[(history[-2][self_index] + history[-1][self_index], history[-2][opp_index] + history[-1][opp_index], internal_memory[268])]
        return 'C' if rnd.random() <= A else 'D'
    elif strategy == 'ZD-M2 / Zero Determinant Memory 2':
        if (len(history) + 1) < 3:
            return 'C'
        A = {('CC', 'CC'): 0.9166666666666666, ('CC', 'CD'): 0.36363636363636365, ('CC', 'DC'): 0.7777777777777778, ('CC', 'DD'): 0.1, ('CD', 'CC'): 0.8333333333333334, ('CD', 'CD'): 0.2727272727272727, ('CD', 'DC'): 0.7777777777777778, ('CD', 'DD'): 0.1, ('DC', 'CC'): 0.6666666666666666, ('DC', 'CD'): 0.09090909090909091, ('DC', 'DC'): 0.7777777777777778, ('DC', 'DD'): 0.1, ('DD', 'CC'): 0.75, ('DD', 'CD'): 0.18181818181818182, ('DD', 'DC'): 0.7777777777777778, ('DD', 'DD'): 0.1}[(history[-2][self_index] + history[-1][self_index], history[-2][opp_index] + history[-1][opp_index])]
        return 'C' if rnd.random() <= A else 'D'
    elif strategy == 'Hard Go By Majority':
        if not len(history): return 'C'
        return choice_majority([history[i][opp_index] for i in range(len(history))], 'D')
    elif strategy == 'Gradual Killer':
        if (len(history)) < 5:
            return 'D'
        elif ((len(history) + 1) - 5) <= 2:
            return 'C'
        elif (len(history) + 1) == 8:
            internal_memory[270] = ((history[-2][opp_index], history[-1][opp_index]) == ('D', 'D'))
        if internal_memory[270]:
            return 'D'
        return 'C'
    elif strategy == 'Forgetful Grudger':        
        maximal_grudging = 10
        if internal_memory[271][1] == maximal_grudging:
            internal_memory[271][1] = 0
            internal_memory[271][0] = False
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[271][0] = True
        if internal_memory[271][0]:
            internal_memory[271][1] += 1
            return 'D'
        return 'C'
    elif strategy == 'Opposite Grudger':
        if len(history) > 0:
            if history[-1][opp_index] == 'C':
                internal_memory[272] = True
        if internal_memory[272]:
            return 'C'
        return 'D'
    elif strategy == 'Aggravater':
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[273] = True
        if (len(history)) < 3:
            return 'D'
        if internal_memory[273]:
            return 'D'
        return 'C'
    elif strategy == 'Soft Grudger':
        if internal_memory[274][0]:
            A = ('D', 'D', 'D', 'C', 'C')[internal_memory[274][1]]
            internal_memory[274][1] += 1
            if internal_memory[274][1] == 5:
                internal_memory[274][1] = 0
                internal_memory[274][0] = False
            return A
        elif len(history) > 0 and history[-1][opp_index] == 'D':
            internal_memory[274][0] = True
            return 'D'
        return 'C'
    elif strategy == 'Grudger Alternator':
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[275] = True
            if internal_memory[275]:
                if history[-1][opp_index] == 'C':
                    return 'D'
        return 'C'
    elif strategy == 'Spiteful CC':
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[276] = True
        if (len(history)) < 2:
            return 'C'
        if internal_memory[276]:
            return 'D'
        return 'C'
    elif strategy == 'Capri':
        hist = [(history[-3][self_index], history[-3][opp_index]) if (len(history) + 1) > 3 else ('C','C'), (history[-2][self_index], history[-2][opp_index]) if (len(history) + 1) > 2 else ('C','C'), (history[-1][self_index], history[-1][opp_index]) if len(history) > 0 else ('C','C')]
        if hist == [('C', 'C'), ('C', 'C'), ('C', 'C')]:  
            return 'C'
        if hist == [('C', 'C'), ('C', 'C'), ('D', 'C')]:  
            return 'C'
        if hist == [('C', 'C'), ('D', 'C'), ('C', 'D')]:
            return 'C'
        if hist == [('D', 'C'), ('C', 'D'), ('C', 'C')]:
            return 'C'
        if hist == [('C', 'D'), ('C', 'C'), ('C', 'C')]:  
            return 'C'
        if hist == [('C', 'C'), ('C', 'C'), ('C', 'D')]:  
            return 'D'
        if hist == [('C', 'C'), ('C', 'D'), ('D', 'C')]:
            return 'C'
        if hist == [('C', 'D'), ('D', 'C'), ('C', 'C')]:
            return 'C'
        if hist == [('D', 'C'), ('C', 'C'), ('C', 'C')]:  
            return 'C'
        if hist == [('D', 'D'), ('D', 'D'), ('D', 'C')]:  
            return 'C'
        if hist == [('D', 'D'), ('D', 'C'), ('C', 'C')]:
            return 'C'
        if hist == [('D', 'D'), ('D', 'D'), ('C', 'D')]:  
            return 'C'
        if hist == [('D', 'D'), ('C', 'D'), ('C', 'C')]:
            return 'C'
        if hist == [('D', 'D'), ('D', 'D'), ('C', 'C')]:  
            return 'C'
        if hist == [('D', 'D'), ('C', 'C'), ('C', 'C')]:
            return 'C'
        return 'D'
    elif strategy == 'Grumpy':
        """A player that gets grumpier the more the opposition defects,
        and nicer the more they cooperate.
        Starts off Nice, but becomes grumpy once the grumpiness threshold is
        hit. Won't become nice once that grumpy threshold is hit, but must
        reach a much lower threshold before it becomes nice again.
        """
        if len(history) > 0:
            if history[-1][opp_index] == 'C':
                internal_memory[277][3] += 1
            else:
                internal_memory[277][4] += 1
        grumpiness = internal_memory[277][4] - internal_memory[277][3]
        if internal_memory[277][0] == "Nice":
            if grumpiness > internal_memory[277][1]:
                internal_memory[277][0] = "Grumpy"
                return 'D'
            return 'C'
        if internal_memory[277][0] == "Grumpy":
            if grumpiness < internal_memory[277][2]:
                internal_memory[277][0] = "Nice"
                return 'C'
            return 'D'
    elif strategy == 'HMM Player':
        def process_hmm_step(current_state, last_opponent_move, t_C, t_D, emissions):
            """
            Menghitung transisi state HMM dan menentukan emisi aksi (C/D).
            """
            probs = t_C[current_state] if last_opponent_move == 'C' else t_D[current_state]
            if 1 in probs:
                next_state = probs.index(1)
            else:
                next_state = int(np.random.choice(len(probs), p=probs))
            p_emit = emissions[next_state]
            if p_emit == 1.0:
                action = 'C'
            elif p_emit == 0.0:
                action = 'D'
            else:
                action = 'C' if np.random.random() < p_emit else 'D'
            return next_state, action
        if not len(history):
            t_C = [[1.0]]
            t_D = [[1.0]]
            emissions = [0.5]
            internal_memory[278] = [0, t_C, t_D, emissions]
            return 'C'
        current_state, t_C, t_D, emissions = internal_memory[278]
        last_move = history[-1][opp_index]
        next_state, action = process_hmm_step(current_state, last_move, t_C, t_D, emissions)
        internal_memory[278][0] = next_state
        return action
    elif strategy == 'Evolved HMM 5':
        def process_hmm_step(current_state, last_opponent_move, t_C, t_D, emissions):
            """
            Menghitung transisi state HMM dan menentukan emisi aksi (C/D).
            """
            probs = t_C[current_state] if last_opponent_move == 'C' else t_D[current_state]
            if 1 in probs:
                next_state = probs.index(1)
            else:
                next_state = int(np.random.choice(len(probs), p=probs))
            p_emit = emissions[next_state]
            if p_emit == 1.0:
                action = 'C'
            elif p_emit == 0.0:
                action = 'D'
            else:
                action = 'C' if np.random.random() < p_emit else 'D'
            return next_state, action
        t_C = [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0.631, 0, 0, 0.369, 0],
            [0.143, 0.018, 0.118, 0, 0.721],
        ]
        t_D = [
            [0, 1, 0, 0, 0],
            [0, 0.487, 0.513, 0, 0],
            [0, 0, 0, 0.590, 0.410],
            [1, 0, 0, 0, 0],
            [0, 0.287, 0.456, 0.146, 0.111],
        ]
        emissions = [1.0, 0.0, 0.0, 1.0, 0.111]
        if not len(history):
            internal_memory[279] = 3
            return 'C'
        current_state = internal_memory[279]
        last_move = history[-1][opp_index]
        next_state, action = process_hmm_step(current_state, last_move, t_C, t_D, emissions)
        internal_memory[279] = next_state
        return action
    elif strategy == 'Eventual Cycle Hunter':
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[282] = True
        if (len(history)) < 10:
            return 'C'
        if (len(history)) % 10 == 0 and internal_memory[282]:
            if cycle_detec(history, opp_index, min_ = 3, offset = 10):
                return 'D'
        return 'C'
    elif strategy == 'Math Constant Hunter':
        """
        Check whether the number of cooperations in the first and second halves
        of the history are close. The variance of the uniform distribution (1/4)
        is a reasonable delta but use something lower for certainty and avoiding
        false positives. This approach will also detect a lot of random players.
        """
        if not len(history):
            return 'C'
        internal_memory[283][(history[-1][opp_index] == 'D')] += 1
        unziped_memory = list(zip(*history))
        self_history, opponent_history = unziped_memory[self_index], unziped_memory[opp_index]
        n = len(history)
        if n >= 8 and internal_memory[283][0] and internal_memory[283][1]:
            start1, end1 = 0, n // 2
            start2, end2 = n // 4, 3 * n // 4
            start3, end3 = n // 2, n
            count1 = opponent_history[start1:end1].count('C') + self_history[
                start1:end1
            ].count('C')
            count2 = opponent_history[start2:end2].count('C') + self_history[
                start2:end2
            ].count('C')
            count3 = opponent_history[start3:end3].count('C') + self_history[
                start3:end3
            ].count('C')
            ratio1 = 0.5 * count1 / (end1 - start1)
            ratio2 = 0.5 * count2 / (end2 - start2)
            ratio3 = 0.5 * count3 / (end3 - start3)
            if abs(ratio1 - ratio2) < 0.2 and abs(ratio1 - ratio3) < 0.2:
                return 'D'
        return 'C'
    elif strategy == 'Random Hunter':
        """
        A random player is unpredictable, which means the conditional frequency
        of cooperation after cooperation, and defection after defections, should
        be close to 50%... although how close is debatable.
        """
        if len(history) > 0:
            internal_memory[284][(history[-1][self_index] == 'D') + 2] += 1
        if (len(history)) > 1:
            if history[-2][self_index] == 'C' and history[-1][opp_index] == 'C':
                internal_memory[284][0] += 1
            if history[-2][self_index] == 'D' and history[-1][opp_index] == 'D':
                internal_memory[284][1] += 1
        n = (len(history))
        if n > 10:
            probabilities = []
            if internal_memory[284][2] > 5:
                probabilities.append(internal_memory[284][0] / internal_memory[284][2])
            if internal_memory[284][3] > 5:
                probabilities.append(internal_memory[284][1] / internal_memory[284][3])
            if probabilities and all(
                [abs(p - 0.5) < 0.25 for p in probabilities]
            ):
                return 'D'
        return 'C'
    elif strategy == 'Inverse':
        if len(history) > 0:
            if internal_memory[285] != None or history[-1][opp_index] == 'D':
                if history[-1][opp_index] == 'C':
                    internal_memory[285] += 1
                else:
                    internal_memory[285] = 1
                return 'C' if rnd.random() <= (1 - (1 / internal_memory[285])) else 'D'
        return 'C'
    elif strategy == 'Winner 12':
        inp = ((history[-1][self_index] if len(history) > 0 else 'C'), (history[-2][opp_index] if (len(history) + 1) > 2 else 'C') + (history[-1][opp_index] if len(history) > 0 else 'C'))
        lookup_table = {('C', 'CC'): 'C', ('C', 'CD'): 'D', ('C', 'DC'): 'C', ('C', 'DD'): 'D', ('D', 'CC'): 'D', ('D', 'CD'): 'C', ('D', 'DC'): 'D', ('D', 'DD'): 'D'}
        return lookup_table[inp]
    elif strategy == 'Winner 21':
        inp = ((history[-1][self_index] if len(history) > 0 else 'C'), (history[-2][opp_index] if (len(history) + 1) > 2 else 'C') + (history[-1][opp_index] if len(history) > 0 else 'C'))
        lookup_table = {('C', 'CC'): 'C', ('C', 'CD'): 'D', ('C', 'DC'): 'C', ('C', 'DD'): 'D', ('D', 'CC'): 'C', ('D', 'CD'): 'D', ('D', 'DC'): 'D', ('D', 'DD'): 'D'}
        return lookup_table[inp]
    elif strategy == 'Phi':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'C':
            internal_memory[286][0] += 1
        else:
            internal_memory[286][1] += 1
        if history[-1][self_index] == 'C':
            internal_memory[286][2] += 1
        else:
            internal_memory[286][3] += 1
        if internal_memory[286][1] == 0:
            return 'D'
        cooperations = internal_memory[286][0] + internal_memory[286][2]
        defections = internal_memory[286][1] + internal_memory[286][3]
        if cooperations / defections > internal_memory[286][4]:
            return 'D'
        return 'C'
    elif strategy == 'Pi':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'C':
            internal_memory[287][0] += 1
        else:
            internal_memory[287][1] += 1
        if history[-1][self_index] == 'C':
            internal_memory[287][2] += 1
        else:
            internal_memory[287][3] += 1
        if internal_memory[287][1] == 0:
            return 'D'
        cooperations = internal_memory[287][0] + internal_memory[287][2]
        defections = internal_memory[287][1] + internal_memory[287][3]
        if cooperations / defections > internal_memory[287][4]:
            return 'D'
        return 'C'
    elif strategy == 'Euler':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'C':
            internal_memory[288][0] += 1
        else:
            internal_memory[288][1] += 1
        if history[-1][self_index] == 'C':
            internal_memory[288][2] += 1
        else:
            internal_memory[288][3] += 1
        if internal_memory[288][1] == 0:
            return 'D'
        cooperations = internal_memory[288][0] + internal_memory[288][2]
        defections = internal_memory[288][1] + internal_memory[288][3]
        if cooperations / defections > internal_memory[288][4]:
            return 'D'
        return 'C'
    elif strategy == 'Generous Tit For Tat Axelrod Project Contributor Team Version':
        if not len(history):
            return 'C'
        A = min(1 - ((self_payoff_matrix['DC'] - self_payoff_matrix['CC']) / (self_payoff_matrix['CC'] - self_payoff_matrix['CD'])), ((self_payoff_matrix['CC'] - self_payoff_matrix['DD']) / (self_payoff_matrix['DC'] - self_payoff_matrix['DD'])))
        if rnd.random() <= A:
            return 'C'
        return history[-1][opp_index]
    elif strategy == 'Firm But Fair / Firm For Tat':
        if not len(history):
            return 'C'
        else:
            if history[-1][opp_index] == history[-1][self_index] and rnd.random() <= (2/3):
                return 'C'
            return history[-1][opp_index]
    elif strategy == 'Stochastic Cooperator':
        return 'C' if rnd.random() <= (0.935, 0.229, 0.266, 0.42)[((2 * (history[-1][self_index] == 'D')) + (history[-1][opp_index] == 'D')) if len(history) > 0 else 0] else 'D'
    elif strategy == 'Stochastic WSLS':
        ep = 0.05
        return 'C' if rnd.random() <= (1.0 - ep, ep, ep, 1.0 - ep)[((2 * (history[-1][self_index] == 'D')) + (history[-1][opp_index] == 'D')) if len(history) > 0 else 0] else 'D'
    elif strategy == 'ALLC OR ALLD':
        if not len(history):
            return 'C' if rnd.random() <= 0.6 else 'D'
        return history[-1][self_index]
    elif strategy == 'Reactive Player':
        if ((history[-1][opp_index] == 'C') if len(history) > 0 else True):
            return 'C' if rnd.random() <= internal_memory[289][0] else 'D'
        return 'C' if rnd.random() <= internal_memory[289][1] else 'D'
    elif strategy == 'MEM2':
        """Actual strategy definition that determines player's action."""
        internal_memory[290][1] -= 1
        if (internal_memory[290][1] == 0) and (internal_memory[290][2] < 2):
            internal_memory[290][1] = 2
            last_two = history[-2:]
            if set(last_two) == {('C', 'C')}:
                internal_memory[290][0] = "TFT"
            elif set(last_two) == {('C', 'D'), ('D', 'C')}:
                internal_memory[290][0] = "TFTT"
            else:
                internal_memory[290][0] = "ALLD"
                internal_memory[290][2] += 1
        if internal_memory[290][0] == "TFT":
            if not len(history):
                return 'C'
            return history[-1][opp_index]
        elif internal_memory[290][0] == 'TFTT':
            if (len(history)) < 2:
                return 'C'
            return 'D' if (history[-2][opp_index], history[-1][opp_index]) == ('D', 'D') else 'C'
        else:
            return 'D'
    elif strategy == 'Memory Decay':
        if not len(history):
            return 'C'
        internal_memory[291].append(history[-1][opp_index])
        if len(history) < 15:
            return history[-1][opp_index]
        t_c = sum(1 for i in internal_memory[291] if i == 'C')
        t_d = len(internal_memory[291]) - t_c
        trust = t_c - (2 * t_d)
        if rnd.random() <= 0.03:
            internal_memory[291][rnd.randint(0, len(internal_memory[291]) - 1)] = 'C' if internal_memory[291][rnd.randint(0, len(internal_memory[291]) - 1)] == 'D' else 'D'
        if rnd.random() <= 0.1:
            internal_memory[291].pop(rnd.randint(0, len(internal_memory[291]) - 1))
        if trust < 0:
            return 'D'
        return 'C'
    elif strategy == 'Meta Winner Ensemble':
        payoff = {('C', 'C'): self_payoff_matrix['CC'],('C', 'D'): self_payoff_matrix['CD'],('D', 'C'): self_payoff_matrix['DC'],('D', 'D'): self_payoff_matrix['DD']}
        meta_win_ens_team = [
            call_strategy('First By Tit For Tat', history, opp_index, opp_name, None, None, None, None, None, None),
            call_strategy('Always Cooperate / Cooperator / Dove / Saint', history, opp_index, opp_name, None, None, None, None, None, None),
            call_strategy('Always Defect / Defector / Hawk / Shark', history, opp_index, opp_name, None, None, None, None, None, None),
            call_strategy('Random', history, opp_index, opp_name, None, None, None, None, None, None),
            call_strategy('First By Grudger / Grim Trigger', history, opp_index, opp_name, internal_memory[292][1], None, None, None, None, None),
            call_strategy('Pavlov / Win-Stay, Lose-Shift', history, opp_index, opp_name, None, None, None, None, None, None),
            call_strategy('Generous Tit For Tat', history, opp_index, opp_name, None, None, None, None, None, None),
            call_strategy('Sample / Second By Tit For 2 Tats', history, opp_index, opp_name, None, None, None, None, None, None),
        ]
        if len(history) > 0:
            internal_memory[292][0] = [internal_memory[292][0][i] + payoff[(internal_memory[292][2][i], history[-1][opp_index])] for i in range(len(internal_memory[292][0]))]
        else:
            internal_memory[292][0] = [0 for _ in range(len(meta_win_ens_team))]
        internal_memory[292][2] = meta_win_ens_team[:]
        prop = max(1, int(len(internal_memory[292][0]) * 0.08))
        best_scores = np.argsort(internal_memory[292][0])[:-prop-1:-1]
        meta_win_ens_majority = sum((1 if meta_win_ens_team[i] == 'C' else -1) for i in range(len(meta_win_ens_team)) if i in best_scores)
        if meta_win_ens_majority < 0:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Meta Hunter':
        meta_hunt_team = [
            call_strategy('Defector Hunter', history, opp_index, opp_name, internal_memory[293], None, None, None, None, None),
            call_strategy('Alternator Hunter', history, opp_index, opp_name, internal_memory[293], None, None, None, None, None),
            call_strategy('Random Hunter', history, opp_index, opp_name, internal_memory[293], None, None, None, None, None),
            call_strategy('Math Constant Hunter', history, opp_index, opp_name, internal_memory[293], None, None, None, None, None),
            call_strategy('Cycle Hunter', history, opp_index, opp_name, None, None, None, None, None, None),
            call_strategy('Eventual Cycle Hunter', history, opp_index, opp_name, internal_memory[293], None, None, None, None, None),
        ]
        if len(history) > 100:
            return history[-1][opp_index]
        return 'D' if 'D' in meta_hunt_team else 'C'
    elif strategy == 'Momentum':
        a = 0.9914655399877477
        threshold = 0.9676595613724907
        if not len(history):
            return 'C'
        internal_memory[294] = (
            (a * internal_memory[294]) + ((1 - a) * (history[-1][opp_index] == 'C'))
        )
        return 'C' if internal_memory[294] >= threshold else 'D'
    elif strategy == 'Desprate':
        if not len(history):
            return rnd.choice(['C','D'])
        return 'C' if history[-1] == ('D','D') else 'D'
    elif strategy == 'Hopeless':
        if not len(history):
            return rnd.choice(['C','D'])
        return 'D' if history[-1] == ('C','C') else 'C'
    elif strategy == 'Willing':
        if not len(history):
            return rnd.choice(['C','D'])
        return 'D' if history[-1] == ('D','D') else 'C'
    elif strategy == 'Negation':
        if not len(history):
            return rnd.choice(['C','D'])
        return 'C' if history[-1][opp_index] == 'D' else 'D'
    elif strategy == 'Once Bitten':
        maximal_grudging = 10
        if internal_memory[295][1] == maximal_grudging:
            internal_memory[295][1] = 0
            internal_memory[295][0] = False
        if (len(history) + 1) > 2:
            if not ('C' in (history[-2][opp_index], history[-1][opp_index])):
                internal_memory[295][0] = True
        if internal_memory[295][0]:
            internal_memory[295][1] += 1
            return 'D'
        return 'C'
    elif strategy == 'Forgetful Fool Me Once':
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[296] += 1
        forget_prop = 0.05
        if rnd.random() <= forget_prop:
            internal_memory[296] = 0
        if internal_memory[296] > 1:
            return 'D'
        else:
            return 'C'
    elif strategy == 'CS / Collective Strategy':
        if len(history) > 0 and history[-1][opp_index] == 'D':
            internal_memory[297] += 1
        if (len(history)) < 2:
            return ('D', 'C')[len(history)]
        if (history[0][opp_index] + history[1][opp_index]) == 'CD':
            return 'C'
        return 'D'
    elif strategy == 'Prober 2':
        if (len(history)) < 3:
            return ('D', 'C', 'C')[(len(history) + 1)-1]
        if (history[1][opp_index] + history[2][opp_index]) == 'DC':
            return 'C'
        else:
            return history[-1][opp_index]
    elif strategy == 'Prober 3':
        if (len(history)) < 2:
            return ('D', 'C')[(len(history) + 1)-1]
        if history[1][opp_index] == 'C':
            return 'D'
        else:
            return history[-1][opp_index]
    elif strategy == 'Prober 4':
        """Actual strategy definition that determines player's action."""
        if not len(history):
            return internal_memory[298][0][0]
        turn = len(history)
        if turn < len(internal_memory[298][0]):
            if history[-1][opp_index] == 'D':
                if history[-1][self_index] == 'D':
                    internal_memory[298][1] += 1
                if history[-1][self_index] == 'C':
                    internal_memory[298][2] += 1
            return internal_memory[298][0][turn]
        if turn == len(internal_memory[298][0]):
            diff_in_Ds = abs(internal_memory[298][1] - internal_memory[298][2])
            internal_memory[298][3] = diff_in_Ds <= 2
        if internal_memory[298][3]:
            return 'D'
        if not internal_memory[298][3]:
            if turn < len(internal_memory[298][0]) + 5:
                return 'C'
            return 'D' if history[-1][opp_index] == 'D' else 'C'
    elif strategy == 'Hard Prober':
        if (len(history)) < 4:
            return ('D', 'D', 'C', 'C')[(len(history) + 1)-1]
        if (history[1][opp_index] + history[2][opp_index]) == 'CC':
            return 'D'
        else:
            return history[-1][opp_index]
    elif strategy == 'Inverse Punisher':
        if len(history) > 0:
            if history[-1][opp_index] == 'C':
                internal_memory[128][3] += 1
        """
        Begins by playing C, then plays D for an amount of rounds proportional
        to the opponents historical '%' of playing D if the opponent ever
        plays D
        """
        if internal_memory[128][2] >= internal_memory[128][0]:
            internal_memory[128][2] = 0
            internal_memory[128][1] = False
        if internal_memory[128][1]:
            internal_memory[128][2] += 1
            return 'D'
        elif ((history[-1][opp_index] == 'D') if len(history) > 0 else False):
            internal_memory[128][0] = (internal_memory[128][3] * 20) // (len(history))
            internal_memory[128][1] = True
            return 'D'
        return 'C'
    elif strategy == 'Level Punisher':
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[146] += 1
        if len(history) < 10:
            return 'C'
        if internal_memory[146] / len(history) > 0.2:
            return 'D'
        return 'C'
    elif strategy == 'Tricky Level Punisher':
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[299] += 1
        else:
            return 'C'
        if len(history) < 10:
            if internal_memory[299] / len(history) > 0.2:
                return 'D'
        if len(history) < 50:
            if internal_memory[299] / len(history) > 0.1:
                return 'D'
        if len(history) < 100:
            if internal_memory[299] / len(history) > 0.05:
                return 'D'
        return 'C'
    elif strategy == 'Double Resurrection':
        if not len(history):
            return 'C'
        if len(history) >= 5:
            inp = [history[-i][self_index] for i in range(1, 5 + 1)] 
            if inp == ['C', 'C', 'C', 'C', 'C']:
                return 'D'
            if inp == ['D', 'D', 'D', 'D', 'D']:
                return 'C'
        return history[-1][opp_index]
    elif strategy == 'Retaliate 2':
        retaliation_threshold = 0.08
        if len(history):
            last_round = (history[-1][self_index], history[-1][opp_index])
            internal_memory[301][last_round] += 1
        CD_count = internal_memory[301][('C', 'D')]
        DC_count = internal_memory[301][('D', 'C')]
        if CD_count > DC_count * retaliation_threshold:
            return 'D'
        return 'C'
    elif strategy == 'Retaliate 3':
        retaliation_threshold = 0.05
        if len(history):
            last_round = (history[-1][self_index], history[-1][opp_index])
            internal_memory[302][last_round] += 1
        CD_count = internal_memory[302][('C', 'D')]
        DC_count = internal_memory[302][('D', 'C')]
        if CD_count > DC_count * retaliation_threshold:
            return 'D'
        return 'C'
    elif strategy == 'Limited Retaliate 2':
        retaliation_threshold = 0.08
        """
        If the opponent has played D to my C more often than x% of the time
        that I've done the same to him, retaliate by playing D but stop doing
        so once I've hit the retaliation limit.
        """
        if len(history):
            last_round = (history[-1][self_index], history[-1][opp_index])
            internal_memory[303][3][last_round] += 1
        CD_count = internal_memory[303][3][('C', 'D')]
        DC_count = internal_memory[303][3][('D', 'C')]
        if CD_count > DC_count * retaliation_threshold:
            internal_memory[303][0] = True
        else:
            internal_memory[303][0] = False
            internal_memory[303][1] = 0
        if internal_memory[303][0]:
            if internal_memory[303][1] < internal_memory[303][2]:
                internal_memory[303][1] += 1
                return 'D'
            else:
                internal_memory[303][1] = 0
                internal_memory[303][0] = False
        return 'C'
    elif strategy == 'Limited Retaliate 3':
        retaliation_threshold = 0.05
        """
        If the opponent has played D to my C more often than x% of the time
        that I've done the same to him, retaliate by playing D but stop doing
        so once I've hit the retaliation limit.
        """
        if len(history):
            last_round = (history[-1][self_index], history[-1][opp_index])
            internal_memory[304][3][last_round] += 1
        CD_count = internal_memory[304][3][('C', 'D')]
        DC_count = internal_memory[304][3][('D', 'C')]
        if CD_count > DC_count * retaliation_threshold:
            internal_memory[304][0] = True
        else:
            internal_memory[304][0] = False
            internal_memory[304][1] = 0
        if internal_memory[304][0]:
            if internal_memory[304][1] < internal_memory[304][2]:
                internal_memory[304][1] += 1
                return 'D'
            else:
                internal_memory[304][1] = 0
                internal_memory[304][0] = False
        return 'C'
    elif strategy == 'First By Downing, But Get Revised By Axelrod':
        round_number = len(history) + 1
        if round_number == 1:
            return 'C'
        if round_number > 2:
            if history[-2][self_index] == 'D':
                if history[-1][opp_index] == 'C':
                    internal_memory[212][3] += 1
                internal_memory[212][5] += 1
                internal_memory[212][1] = internal_memory[212][3] / internal_memory[212][5]
            else:
                if history[-1][opp_index] == 'C':
                    internal_memory[212][2] += 1
                internal_memory[212][4] += 1
                internal_memory[212][0] = internal_memory[212][2] / internal_memory[212][4]
        c = 6.0 * internal_memory[212][0] - 8.0 * internal_memory[212][1] - 2
        alt = 4.0 * internal_memory[212][0] - 5.0 * internal_memory[212][1] - 1
        if c >= 0 and c >= alt:
            move = 'C'
        elif (0 <= c < alt) or (alt >= 0):
            move = flip(history[-1][self_index])
        else:
            move = 'D'
        return move
    elif strategy == 'Self Steem':
        """Actual strategy definition that determines player's action."""
        turns_number = len(history)
        sine_value = mth.sin(2 * mth.pi * turns_number / 10)
        if sine_value > 0.95:
            return 'D'
        if 0.95 > abs(sine_value) > 0.3:
            return history[-1][opp_index]
        if 0.3 > sine_value > -0.3:
            return rnd_prob(0.5)
        return 'C'
    elif strategy == 'Thue Morse':
        if not len(history):
            return 'C'
        else:
            if internal_memory[305][1] == 0:
                thuemorse_code = internal_memory[305][0]
                for i in internal_memory[305][0]:
                    if i == '0':
                        thuemorse_code += '1'
                    else:
                        thuemorse_code += '0'
                internal_memory[305][0] = thuemorse_code
                internal_memory[305][1] = len(thuemorse_code)
            internal_memory[305][1] -= 1
            return 'C' if internal_memory[305][0][len(internal_memory[305][0]) - (internal_memory[305][1] + 1)] == '1' else 'D'
    elif strategy == 'Short Mem':
        if len(history) <= 10:
            return 'C'
        array = [history[-i][opp_index] for i in range(1, 10 + 1)] 
        C_counts = array.count('C')
        D_counts = array.count('D')
        if C_counts - D_counts >= 3:
            return 'C'
        elif D_counts - C_counts >= 3:
            return 'D'
        else:
            return history[-1][opp_index]
    elif strategy == 'Stalker':
        very_good_score = self_payoff_matrix['CC']
        very_bad_score = self_payoff_matrix['DD']
        wish_score = (self_payoff_matrix['CC'] + self_payoff_matrix['DD']) / 2
        """Actual strategy definition that determines player's action."""
        if not len(history):
            return 'C'
        current_average_score = score[self_index] / len(history)
        if current_average_score > very_good_score:
            return 'D'
        if (current_average_score > wish_score) and (
            current_average_score < very_good_score
        ):
            return 'C'
        if current_average_score > 2:
            return 'C'
        if (current_average_score < 2) and (current_average_score > 1):
            return 'D'
        return rnd_prob(0.5)
    elif strategy == 'Dynamic 2 Tits For Tat':
        """Actual strategy definition that determines player's action."""
        if not len(history):
            return 'C'
        internal_memory[306] += (history[-1][opp_index] == 'C')
        inp = [history[-i][opp_index] for i in range(1, min(2, len(history)) + 1)] 
        if 'D' in inp:
            return 'C'  if rnd.random() <= (
                internal_memory[306] / len(history)
            ) else 'D'
        else:
            return 'C'
    elif strategy == 'Hard Tit For Tat':
        if not len(history):
            return 'C'
        inp = [history[-i][opp_index] for i in range(1, min(3, len(history)) + 1)] 
        if 'D' in inp:
            return 'D'
        return 'C'
    elif strategy == 'Hard Tit For 2 Tats':
        if not len(history):
            return 'C'
        inp = [history[-i][opp_index] for i in range(1, min(3, len(history)) + 1)] 
        if 'DD' in "".join(inp):
            return 'D'
        return 'C'
    elif strategy == 'Omega Tit For Tat':
        """Actual strategy definition that determines player's action."""
        if not len(history):
            return 'C'
        if len(history) == 1:
            return history[-1][opp_index]
        if internal_memory[307][3] >= internal_memory[307][0]:
            move = 'C'
            if internal_memory[307][3] == internal_memory[307][0]:
                internal_memory[307][3] = internal_memory[307][0] + 1
            else:
                internal_memory[307][3] = 0
        else:
            if len(history) >= 2:
                if (history[-2][opp_index], history[-1][opp_index]) == ('C', 'C'):
                    internal_memory[307][2] -= 1
            if history[-2][opp_index] != history[-1][opp_index]:
                internal_memory[307][2] += 1
            if history[-1][self_index] != history[-1][opp_index]:
                internal_memory[307][2] += 1
            if internal_memory[307][2] >= internal_memory[307][1]:
                move = 'D'
            else:
                move = history[-1][opp_index]
                if history[-2][opp_index] != history[-1][opp_index]:
                    internal_memory[307][3] += 1
                else:
                    internal_memory[307][3] = 0
        return move
    elif strategy == 'Gradual Cristal Version':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'D':
            internal_memory[281][1] += 1
        if internal_memory[281][2] > 0:
            internal_memory[281][2] -= 1
            return 'C'
        if history[-1][opp_index] == 'D' or internal_memory[281][0] > 0:
            if history[-1][opp_index] == 'D':
                if internal_memory[281][0] == 0:
                    internal_memory[281][0] = internal_memory[281][1]
            internal_memory[281][0] -= 1
            if internal_memory[281][0] == 0:
                internal_memory[281][2] = 2
            return 'D'
        else:
            return 'C'
    elif strategy == 'Spiteful Tit For Tat':
        if not len(history):
            return 'C'
        if len(history) >= 2:
            if ((history[-2][opp_index], history[-1][opp_index]) == ('D', 'D')) or internal_memory[308]:
                internal_memory[308] = True
                return 'D'
        return history[-1][opp_index]
    elif strategy == 'Slow Tit For Tat':
        """Actual strategy definition that determines player's action."""
        if len(history) < 2:
            return 'C'
        if history[-2][opp_index] == history[-1][opp_index]:
            return history[-1][opp_index]
        return history[-1][self_index]
    elif strategy == 'Alexei / End Game Cheater':
        if not len(history):
            return 'C'
        if len(history) >= (tournament_avg_last_round):
            return 'D'
        return history[-1][opp_index]
    elif strategy == 'Eugine Nier':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'D':
            internal_memory[309] += 1
        if (internal_memory[309] >= 5) or (len(history) >= (tournament_avg_last_round)):
            return 'D'
        return history[-1][opp_index]
    elif strategy == 'N Tits For M Tats':
        if len(history) <= (internal_memory[310][0] - 1):
            return  'C'
        inp = sum([(history[-i][opp_index] == 'D') for i in range(1, internal_memory[310][0] + 1)]) 
        if inp == internal_memory[310][0]:
            internal_memory[310][2] = internal_memory[310][1]
        if internal_memory[310][2] > 0:
            internal_memory[310][2] -= 1
            return 'D'
        return 'C'
    elif strategy == 'Michaelos':
        """Actual strategy definition that determines player's action."""
        if not len(history):
            return 'C'
        if len(history) >= (tournament_avg_last_round):
            internal_memory[311] = True
        if internal_memory[311]:
            return 'D'
        if history[-1][self_index] == 'D' and history[-1][opp_index] == 'C':
            decision = rnd_prob(0.5)
            if decision == 'C':
                return 'C'
            else:
                internal_memory[311] = True
                return 'D'
        return history[-1][opp_index]
    elif strategy == 'Random Tit For Tat':
        Peta = 0.5
        if len(history) % 2 == 0:
            return history[-1][opp_index] if len(history) > 0 else 'C'
        return 'C' if rnd.random() <= Peta else 'D'
    elif strategy == 'BBE / Burn Both Ends':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'C':
            return 'C' if rnd.random() <= 0.9 else 'D'
        return 'D'
    elif strategy == 'Very Bad':
        """Actual strategy definition that determines player's action."""
        total_moves = len(history)
        if total_moves > 1 and history[-1][opp_index] == 'C':
            internal_memory[312] += 1
        if total_moves < 3:
            return 'C'
        cooperations = internal_memory[312]
        cooperation_probability = cooperations / total_moves
        if cooperation_probability > 0.5:
            return 'C'
        elif cooperation_probability < 0.5:
            return 'D'
        else:
            return history[-1][opp_index]
    elif strategy == 'Knowledgeable Worse & Worse':
        return 'D' if rnd.random() <= (len(history) / tournament_avg_last_round) else 'C'
    elif strategy == 'Worse & Worse 2':
        current_round = len(history) + 1
        if current_round == 1:
            return 'C'
        elif current_round <= 20:
            return history[-1][opp_index]
        else:
            probability = 20 / current_round
            return 'C' if rnd.random() <= probability else 'D'
    elif strategy == 'ZD-X Lars Roemheld Version / Zero Determinant Extortion Lars Roemheld Version':
        if not len(history):
            return 'C'
        phi = 0.2
        s = 0.1
        l = 1.0
        clamped_l = max(min(l, self_payoff_matrix['CC']), self_payoff_matrix['DD'])
        s_min = -min((self_payoff_matrix['DC'] - clamped_l) / (clamped_l - self_payoff_matrix['CD']), (clamped_l - self_payoff_matrix['CD']) / (self_payoff_matrix['DC'] - clamped_l))
        phi = max(min(phi, 1), 0)
        s = max(min(s, 1), s_min)
        l = clamped_l
        p1 = 1 - phi * (1 - s) * (self_payoff_matrix['CC'] - l)
        p2 = 1 - phi * (s * (l - self_payoff_matrix['CD']) + (self_payoff_matrix['DC'] - l))
        p3 = phi * ((l - self_payoff_matrix['CD']) + s * (self_payoff_matrix['DC'] - l))
        p4 = phi * (1 - s) * (l - self_payoff_matrix['DD'])
        four_vector = [p1, p2, p3, p4]
        return 'C' if rnd.random() <= four_vector[((2 * (history[-1][self_index] == 'D')) + (history[-1][opp_index] == 'D'))] else 'D'
    elif strategy == 'ZD-G / Zero Determinant Generous':
        if not len(history):
            return 'C'
        phi = 1/8
        s = 0.5
        l = 3.0
        clamped_l = max(min(l, self_payoff_matrix['CC']), self_payoff_matrix['DD'])
        s_min = -min((self_payoff_matrix['DC'] - clamped_l) / (clamped_l - self_payoff_matrix['CD']), (clamped_l - self_payoff_matrix['CD']) / (self_payoff_matrix['DC'] - clamped_l))
        phi = max(min(phi, 1), 0)
        s = max(min(s, 1), s_min)
        l = clamped_l
        p1 = 1 - phi * (1 - s) * (self_payoff_matrix['CC'] - l)
        p2 = 1 - phi * (s * (l - self_payoff_matrix['CD']) + (self_payoff_matrix['DC'] - l))
        p3 = phi * ((l - self_payoff_matrix['CD']) + s * (self_payoff_matrix['DC'] - l))
        p4 = phi * (1 - s) * (l - self_payoff_matrix['DD'])
        four_vector = [p1, p2, p3, p4]
        return 'C' if rnd.random() <= four_vector[((2 * (history[-1][self_index] == 'D')) + (history[-1][opp_index] == 'D'))] else 'D'
    elif strategy == 'ZD-GTFT / Zero Determinant Generous Tit For Tat':
        if not len(history):
            return 'C'
        phi = 0.25
        s = 0.5
        l = 3.0
        clamped_l = max(min(l, self_payoff_matrix['CC']), self_payoff_matrix['DD'])
        s_min = -min((self_payoff_matrix['DC'] - clamped_l) / (clamped_l - self_payoff_matrix['CD']), (clamped_l - self_payoff_matrix['CD']) / (self_payoff_matrix['DC'] - clamped_l))
        phi = max(min(phi, 1), 0)
        s = max(min(s, 1), s_min)
        l = clamped_l
        p1 = 1 - phi * (1 - s) * (self_payoff_matrix['CC'] - l)
        p2 = 1 - phi * (s * (l - self_payoff_matrix['CD']) + (self_payoff_matrix['DC'] - l))
        p3 = phi * ((l - self_payoff_matrix['CD']) + s * (self_payoff_matrix['DC'] - l))
        p4 = phi * (1 - s) * (l - self_payoff_matrix['DD'])
        four_vector = [p1, p2, p3, p4]
        return 'C' if rnd.random() <= four_vector[((2 * (history[-1][self_index] == 'D')) + (history[-1][opp_index] == 'D'))] else 'D'
    elif strategy == 'ZD-MS / Zero Determinant Mischief':
        if not len(history):
            return 'C'
        phi = 0.1
        s = 0.0
        l = 1.0
        clamped_l = max(min(l, self_payoff_matrix['CC']), self_payoff_matrix['DD'])
        s_min = -min((self_payoff_matrix['DC'] - clamped_l) / (clamped_l - self_payoff_matrix['CD']), (clamped_l - self_payoff_matrix['CD']) / (self_payoff_matrix['DC'] - clamped_l))
        phi = max(min(phi, 1), 0)
        s = max(min(s, 1), s_min)
        l = clamped_l
        p1 = 1 - phi * (1 - s) * (self_payoff_matrix['CC'] - l)
        p2 = 1 - phi * (s * (l - self_payoff_matrix['CD']) + (self_payoff_matrix['DC'] - l))
        p3 = phi * ((l - self_payoff_matrix['CD']) + s * (self_payoff_matrix['DC'] - l))
        p4 = phi * (1 - s) * (l - self_payoff_matrix['DD'])
        four_vector = [p1, p2, p3, p4]
        return 'C' if rnd.random() <= four_vector[((2 * (history[-1][self_index] == 'D')) + (history[-1][opp_index] == 'D'))] else 'D'
    elif strategy == 'ZD-S / Zero Determinant Set':
        if not len(history):
            return 'C'
        phi = 1/4
        s = 0.0
        l = 2.0
        clamped_l = max(min(l, self_payoff_matrix['CC']), self_payoff_matrix['DD'])
        s_min = -min((self_payoff_matrix['DC'] - clamped_l) / (clamped_l - self_payoff_matrix['CD']), (clamped_l - self_payoff_matrix['CD']) / (self_payoff_matrix['DC'] - clamped_l))
        phi = max(min(phi, 1), 0)
        s = max(min(s, 1), s_min)
        l = clamped_l
        p1 = 1 - phi * (1 - s) * (self_payoff_matrix['CC'] - l)
        p2 = 1 - phi * (s * (l - self_payoff_matrix['CD']) + (self_payoff_matrix['DC'] - l))
        p3 = phi * ((l - self_payoff_matrix['CD']) + s * (self_payoff_matrix['DC'] - l))
        p4 = phi * (1 - s) * (l - self_payoff_matrix['DD'])
        four_vector = [p1, p2, p3, p4]
        return 'C' if rnd.random() <= four_vector[((2 * (history[-1][self_index] == 'D')) + (history[-1][opp_index] == 'D'))] else 'D'
    elif strategy == 'ZD-Q / Zero Determinant Equalizer':
        if not len(history):
            return 'C'
        phi = 0.125
        s = 0.0
        l = 2.0
        clamped_l = max(min(l, self_payoff_matrix['CC']), self_payoff_matrix['DD'])
        s_min = -min((self_payoff_matrix['DC'] - clamped_l) / (clamped_l - self_payoff_matrix['CD']), (clamped_l - self_payoff_matrix['CD']) / (self_payoff_matrix['DC'] - clamped_l))
        phi = max(min(phi, 1), 0)
        s = max(min(s, 1), s_min)
        l = clamped_l
        p1 = 1 - phi * (1 - s) * (self_payoff_matrix['CC'] - l)
        p2 = 1 - phi * (s * (l - self_payoff_matrix['CD']) + (self_payoff_matrix['DC'] - l))
        p3 = phi * ((l - self_payoff_matrix['CD']) + s * (self_payoff_matrix['DC'] - l))
        p4 = phi * (1 - s) * (l - self_payoff_matrix['DD'])
        four_vector = [p1, p2, p3, p4]
        return 'C' if rnd.random() <= four_vector[((2 * (history[-1][self_index] == 'D')) + (history[-1][opp_index] == 'D'))] else 'D'
    elif strategy == 'Laran':
        if not len(history):
            return "C"
        otak = internal_memory[314]
        last_my = history[-1][self_index]
        last_opp = history[-1][opp_index]
        curr = otak["current_state"]
        for s in range(otak["num_states"]):
            if otak["state_outputs"][s] == last_opp:
                otak["transitions"][curr][last_my][s] = 1.0  
                break
        probs = otak["transitions"][curr][last_my]
        otak["current_state"] = probs.index(max(probs))
        predicted_opp = otak["state_outputs"][otak["current_state"]]
        return "D" if predicted_opp == "C" else "C"
    elif strategy == 'Turan':
        if not len(history):
            return "C"
        otak = internal_memory[315]
        last_my = history[-1][self_index]
        last_opp = history[-1][opp_index]
        curr = otak["current_state"]
        probs = otak["transitions"][curr][last_my]
        for s in range(otak["num_states"]):
            if otak["state_outputs"][s] == last_opp:
                probs[s] += 0.25  
        total = sum(probs)
        otak["transitions"][curr][last_my] = [p / total for p in probs]
        otak["current_state"] = probs.index(max(probs))
        predicted_opp = otak["state_outputs"][otak["current_state"]]
        return "D" if predicted_opp == "C" else "C"
    elif strategy == 'Tages':
        if not len(history):
            return "C"
        otak = internal_memory[316]
        look_ahead_depth = 10
        payoff = {('C', 'C'): self_payoff_matrix['CC'],('C', 'D'): self_payoff_matrix['CD'],('D', 'C'): self_payoff_matrix['DC'],('D', 'D'): self_payoff_matrix['DD']}
        last_my = history[-1][self_index]
        last_opp = history[-1][opp_index]
        curr = otak["current_state"]
        probs = otak["transitions"][curr][last_my]
        for s in range(otak["num_states"]):
            if otak["state_outputs"][s] == last_opp:
                probs[s] += 0.25
        total = sum(probs)
        otak["transitions"][curr][last_my] = [p / total for p in probs]
        otak["current_state"] = probs.index(max(probs))
        def _simulasi(mulai_state, aksi_awal):
            skor_total = 0
            s_shadow = mulai_state
            a_shadow = aksi_awal
            for _ in range(look_ahead_depth):
                p_shadow = otak["transitions"][s_shadow][a_shadow]
                s_shadow = rnd.choices(range(otak["num_states"]), weights=p_shadow)[0]
                pred_opp = otak["state_outputs"][s_shadow]
                skor_total += payoff[(a_shadow, pred_opp)]
                a_shadow = pred_opp
            return skor_total
        skor_C = _simulasi(otak["current_state"], "C")
        skor_D = _simulasi(otak["current_state"], "D")
        return "C" if skor_C >= skor_D else "D"
    elif strategy == 'Pavlov D / Suspicious Pavlov':
        if not len(history):
            return 'D'
        return history[-1][self_index] if self_payoff_matrix[history[-1][self_index] + history[-1][opp_index]] >= self_avg_payoff_matrix else flip(history[-1][self_index])
    elif strategy == 'Free Rider':
        if len(history) > 0 and history[-1][opp_index] == 'D':
            return '[OUT]'
        return 'D'
    elif strategy == 'Rover':
        threshold = 2.0
        if len(history) > 0 and (score[self_index] - internal_memory[318]) < threshold:
            return '[OUT]'
        internal_memory[318] = score[self_index]
        return 'D'
    elif strategy == 'Vengeful':
        if not len(history):
            return 'C'
        instruction_string = "1:C1|D2,2:D3|D3,3:C1|D2"
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                mode_number, action = item.split(':')
                action_0_str, action_1_str = action.split('|')
                action_0 = [action_0_str[0], action_0_str[1:]]
                action_1 = [action_1_str[0], action_1_str[1:]]
                rule[mode_number] = [action_0, action_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[internal_memory[319]][(history[-1][opp_index] == 'D')]
        internal_memory[319] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'Deadlock Breaker':
        if not len(history):
            return 'C'
        instruction_string = "1:C1|D2,2:C1|D3,3:C1|C1"
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                mode_number, action = item.split(':')
                action_0_str, action_1_str = action.split('|')
                action_0 = [action_0_str[0], action_0_str[1:]]
                action_1 = [action_1_str[0], action_1_str[1:]]
                rule[mode_number] = [action_0, action_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[internal_memory[320]][(history[-1][opp_index] == 'D')]
        internal_memory[320] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'Sugar Strategy / Treasure Hunt':
        if not len(history):
            return 'C'
        instruction_string = "1:C2|D4,2:D3|D4,3:C3|D4,4:D4|D4"
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                mode_number, action = item.split(':')
                action_0_str, action_1_str = action.split('|')
                action_0 = [action_0_str[0], action_0_str[1:]]
                action_1 = [action_1_str[0], action_1_str[1:]]
                rule[mode_number] = [action_0, action_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[internal_memory[321]][(history[-1][opp_index] == 'D')]
        internal_memory[321] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'Nasty Tit For Tat':
        if rnd.random() <= 0.1:
            internal_memory[322] = 2
        if internal_memory[322] > 0:
            internal_memory[322] -= 1
            return 'D'
        return history[-1][opp_index] if len(history) > 0 else 'C'
    elif strategy == 'Analogy':
        if len(history) < 2:
            return 'C'
        if history[-2][self_index] == 'C':
            internal_memory[323][0] += 1
            if history[-1][opp_index] == 'C':
                internal_memory[323][1] += 1
        else:
            internal_memory[323][2] += 1
            if history[-1][opp_index] == 'C':
                internal_memory[323][3] += 1
        pc_given_c = (internal_memory[323][1] / internal_memory[323][0]) if internal_memory[323][0] > 0 else 0.5
        pc_given_d = (internal_memory[323][3] / internal_memory[323][2]) if internal_memory[323][2] > 0 else 0.5
        E_c = (pc_given_c * self_payoff_matrix['CC']) + ((1 - pc_given_c) * self_payoff_matrix['CD'])
        E_d = (pc_given_d * self_payoff_matrix['DC']) + ((1 - pc_given_d) * self_payoff_matrix['DD'])
        return 'C' if E_c > E_d else 'D'
    elif strategy == 'Look Ahead / Look Up':
        depth = 5
        payoff = {
            ('C', 'C'): (self_payoff_matrix['CC'], self_payoff_matrix['CC']),
            ('C', 'D'): (self_payoff_matrix['CD'], self_payoff_matrix['DC']),
            ('D', 'C'): (self_payoff_matrix['DC'], self_payoff_matrix['CD']),
            ('D', 'D'): (self_payoff_matrix['DD'], self_payoff_matrix['DD'])
        }
        if len(history) < 2:
            return 'C'
        def predict_opponent_response(last_my_move, last_opp_move):
            count_C = 0
            count_D = 0
            for i in range(len(history) - 1):
                if history[i][self_index] == last_my_move and history[i][opp_index] == last_opp_move:
                    if history[i + 1][opp_index] == 'C':
                        count_C += 1
                    else:
                        count_D += 1
            if count_C == 0 and count_D == 0:
                return last_opp_move  
            return 'C' if count_C >= count_D else 'D'
        def simulate_future_score(chosen_action):
            total_score = 0
            pred_opp = predict_opponent_response(history[-1][self_index], history[-1][opp_index])
            total_score += payoff[(chosen_action, pred_opp)][0]
            curr_my = chosen_action
            curr_opp = pred_opp
            for _ in range(1, depth):
                next_my = 'C' if curr_opp == 'C' else 'D'
                next_opp = predict_opponent_response(curr_my, curr_opp)
                total_score += payoff[(next_my, next_opp)][0]
                curr_my, curr_opp = next_my, next_opp
            return total_score
        score_if_C = simulate_future_score('C')
        score_if_D = simulate_future_score('D')
        return 'C' if score_if_C >= score_if_D else 'D'
    elif strategy == 'Tit For 3 Tats':
        if len(history) >= 3:
            if (history[-3][opp_index], history[-2][opp_index], history[-1][opp_index]) == ('D', 'D', 'D'):
                return 'D'
        return 'C'
    elif strategy == 'Lenient Grim 2':
        if not len(history):
            return 'C'
        if len(history) >= 2:
            if (history[-2][opp_index], history[-1][opp_index]) == ('D', 'D'):
                return 'D'
        return history[-1][self_index]
    elif strategy == 'Lenient Grim 3':
        if not len(history):
            return 'C'
        if len(history) >= 3:
            if (history[-3][opp_index], history[-2][opp_index], history[-1][opp_index]) == ('D', 'D', 'D'):
                return 'D'
        return history[-1][self_index]
    elif strategy == 'Exploratory Tit For 3 Tats':
        if len(history) >= 3:
            if (history[-3][opp_index], history[-2][opp_index], history[-1][opp_index]) == ('D', 'D', 'D'):
                return 'D'
        return 'C' if rnd.random() > 0.1 else 'D'
    elif strategy == 'Exploratory Lenient Grim 2':
        if not len(history):
            return 'C' if rnd.random() > 0.1 else 'D'
        if len(history) >= 2:
            if (history[-2][opp_index], history[-1][opp_index]) == ('D', 'D'):
                internal_memory[324] = True
        if internal_memory[324]:
            return 'D'
        else:
            return 'C' if rnd.random() > 0.1 else 'D'
    elif strategy == 'False Cooperator':
        return 'C' if len(history) < 10 else 'D'
    elif strategy == 'Boxer':
        d_threshold = 0.4
        min_rounds = 3
        req_c_to_recover = 3
        if not len(history):
            return 'C'
        internal_memory[325][1].append(history[-1][opp_index])
        if internal_memory[325][0]:
            recent_opp_moves = internal_memory[325][1][-req_c_to_recover:]
            if len(recent_opp_moves) == req_c_to_recover and all(m == 'C' for m in recent_opp_moves):
                internal_memory[325][0] = False  
                return 'C'
            return 'D'
        if len(internal_memory[325][1]) >= min_rounds:
            d_count = internal_memory[325][1].count('D')
            d_ratio = d_count / len(internal_memory[325][1])
            if d_ratio > d_threshold:
                internal_memory[325][0] = True
                return 'D'
        return 'C'
    elif strategy == 'Bros Mind':
        decay_rate = 0.7  
        if not len(history):
            return 'C'
        last_opp_move = history[-1][opp_index]
        if last_opp_move == 'C':
            internal_memory[326] = min(1.0, internal_memory[326] + 0.1)
        else:
            internal_memory[326] *= decay_rate
        return 'C' if internal_memory[326] >= 0.5 else 'D'
    elif strategy == 'Crabby':
        anger_threshold = 2
        punishment_length = 3
        if not len(history):
            return 'C'
        if internal_memory[327] > 0:
            internal_memory[327] -= 1
            return 'D'
        recent_opp = [history[-i][opp_index] for i in range(1, min(3, len(history)) + 1)] 
        if recent_opp.count('D') >= anger_threshold:
            internal_memory[327] = punishment_length - 1
            return 'D'
        return 'C'
    elif strategy == 'Observant':
        if len(history) < 5:
            return 'C'
        inp = [history[-i][opp_index] for i in range(1, len(history) + 1)] 
        d_ratio = inp.count('D') / len(history)
        if d_ratio > 0.5:
            return 'D'
        return 'C'
    elif strategy == 'Mensa':
        if len(history) < 3:
            return 'C'
        inp = [history[-i][opp_index] for i in range(1, len(history) + 1)] 
        c_ratio = inp.count('C') / len(history)
        if c_ratio > 0.8:
            return 'D'
        return history[-1][opp_index]
    elif strategy == 'Killer':
        if len(history) < 4:
            return 'D' if len(history) % 2 == 1 else 'C'
        inp = [history[-i][opp_index] for i in range(1, len(history) + 1)] 
        if inp.count('D') == 0:
            return 'D'
        return history[-1][opp_index]
    elif strategy == 'MARS / Mimicry And Relative Similarity':
        window_size = 5
        threshold = 0.6
        if len(history) < window_size:
            return 'C'  
        matches = sum(1 for m, o in history[-window_size:] if m == o)
        similarity = matches / window_size
        if similarity >= threshold:
            return 'C'
        return history[-1][opp_index]
    elif strategy == 'Bandit UCB':
        payoff = {
            ('C', 'C'): self_payoff_matrix['CC'], ('C', 'D'): self_payoff_matrix['CD'],
            ('D', 'C'): self_payoff_matrix['DC'], ('D', 'D'): self_payoff_matrix['DD']
        }
        def _update_payoffs():
            if not len(history):
                return
            last_my = history[-1][self_index]
            last_opp = history[-1][opp_index]
            reward = payoff[(last_my, last_opp)]
            internal_memory[328][0][last_my] += 1
            internal_memory[328][1][last_my] += reward
        _update_payoffs()
        t = len(history)
        if t < 2:
            return 'C' if t == 0 else 'D'
        ucb_values = {}
        for action in ['C', 'D']:
            mean_reward = internal_memory[328][1][action] / internal_memory[328][0][action]
            bonus = mth.sqrt((2 * mth.log(t)) / internal_memory[328][0][action])
            ucb_values[action] = mean_reward + bonus
        if ucb_values['C'] == ucb_values['D']:
            return rnd_prob(0.5)
        return 'C' if ucb_values['C'] > ucb_values['D'] else 'D'
    elif strategy == 'MCMC / Markov Chain Monte Carlo':
        num_samples = 100
        if len(history) < 2:
            return 'C'
        current_state = (history[-1][self_index], history[-1][opp_index])
        matching_next_opp_moves = []
        for i in range(len(history) - 1):
            state_i = (history[i][self_index], history[i][opp_index])
            if state_i == current_state:
                matching_next_opp_moves.append(history[i+1][opp_index])
        if not matching_next_opp_moves:
            return history[-1][opp_index]  
        d_count = 0
        for _ in range(num_samples):
            sampled_move = rnd.choice(matching_next_opp_moves)
            if sampled_move == 'D':
                d_count += 1
        prob_opp_defect = d_count / num_samples
        if prob_opp_defect > 0.5:
            return 'D'
        return 'C'
    elif strategy == 'Star S':
        handshake = ['C', 'C', 'D', 'C', 'D']
        if len(history) < len(handshake):
            return handshake[len(history)]
        inp = [history[i][opp_index] for i in range(len(handshake))]
        if inp == handshake:
            return 'D'
        return 'D'
    elif strategy == 'Star SL':
        handshake = ['C', 'C', 'D', 'C', 'D']
        if len(history) < len(handshake):
            return handshake[len(history)]
        inp = [history[i][opp_index] for i in range(len(handshake))]
        if inp == handshake:
            return 'D'
        return history[-1][opp_index]
    elif strategy == 'Star SN':
        handshake = ['C', 'C', 'D', 'C', 'D']
        max_noise_error = 1
        if len(history) < len(handshake):
            return handshake[len(history)]
        inp = [history[i][opp_index] for i in range(len(handshake))]
        verify = sum((inp[i] != handshake[i]) for i in range(len(handshake)))
        if verify <= max_noise_error:
            return 'D'
        return 'D'
    elif strategy == 'Star Slave 1':
        handshake = ['C', 'C', 'D', 'C', 'D']
        if len(history) < len(handshake):
            return handshake[len(history)]
        inp = [history[i][opp_index] for i in range(len(handshake))]
        if inp == handshake:
            return 'C'
        return 'D'
    elif strategy == 'Star Slave 2':
        handshake = ['C', 'C', 'D', 'C', 'D']
        if len(history) < len(handshake):
            return handshake[len(history)]
        inp = [history[i][opp_index] for i in range(len(handshake))]
        if inp == handshake:
            return 'C'
        return 'D'
    elif strategy == 'Star Slave 3':
        handshake = ['C', 'C', 'D', 'C', 'D']
        if len(history) < len(handshake):
            return handshake[len(history)]
        inp = [history[i][opp_index] for i in range(len(handshake))]
        if inp == handshake:
            return 'C'
        return 'D'
    elif strategy == 'Star Slave 4':
        handshake = ['C', 'C', 'D', 'C', 'D']
        if len(history) < len(handshake):
            return handshake[len(history)]
        inp = [history[i][opp_index] for i in range(len(handshake))]
        if inp == handshake:
            return 'C'
        return 'D'
    elif strategy == 'Star Slave 5':
        handshake = ['C', 'C', 'D', 'C', 'D']
        if len(history) < len(handshake):
            return handshake[len(history)]
        inp = [history[i][opp_index] for i in range(len(handshake))]
        if inp == handshake:
            return 'C'
        return 'D'
    elif strategy == 'Star Slave 6':
        handshake = ['C', 'C', 'D', 'C', 'D']
        if len(history) < len(handshake):
            return handshake[len(history)]
        inp = [history[i][opp_index] for i in range(len(handshake))]
        if inp == handshake:
            return 'C'
        return 'D'
    elif strategy == 'Star Slave 7':
        handshake = ['C', 'C', 'D', 'C', 'D']
        if len(history) < len(handshake):
            return handshake[len(history)]
        inp = [history[i][opp_index] for i in range(len(handshake))]
        if inp == handshake:
            return 'C'
        return 'D'
    elif strategy == 'Star Slave 8':
        handshake = ['C', 'C', 'D', 'C', 'D']
        if len(history) < len(handshake):
            return handshake[len(history)]
        inp = [history[i][opp_index] for i in range(len(handshake))]
        if inp == handshake:
            return 'C'
        return 'D'
    elif strategy == 'Star Slave 9':
        handshake = ['C', 'C', 'D', 'C', 'D']
        if len(history) < len(handshake):
            return handshake[len(history)]
        inp = [history[i][opp_index] for i in range(len(handshake))]
        if inp == handshake:
            return 'C'
        return 'D'
    elif strategy == 'Star Slave 10':
        handshake = ['C', 'C', 'D', 'C', 'D']
        if len(history) < len(handshake):
            return handshake[len(history)]
        inp = [history[i][opp_index] for i in range(len(handshake))]
        if inp == handshake:
            return 'C'
        return 'D'
    elif strategy == 'Generous Tit For Tat Less Wrong Version':
        if not len(history):
            return 'C'
        if rnd.random() <= 0.0000004839:
            return 'C'
        return history[-1][opp_index]
    elif strategy == 'Caerbannog':
        if not len(history):
            return 'C'
        if len(history) >= (tournament_avg_last_round):
            return 'D'
        if rnd.random() <= 0.2:
            return 'C'
        return history[-1][opp_index]
    elif strategy == 'Jem':
        if len(history) >= 2:
            if history[-1][opp_index] == 'D':
                if history[-2][self_index] == 'C':
                    internal_memory[329] += 1
        if len(history) >= (tournament_avg_last_round):
            return 'D'
        A = 0.5 ** internal_memory[329]
        if rnd.random() <= A:
            return 'C'
        return history[-1][opp_index]
    elif strategy == 'Rwallace':
        internal_memory[330] += (history[-1][opp_index] == 'D') if len(history) > 0 else 0
        if (len(history) >= (tournament_avg_last_round)) or (internal_memory[330] >= 3):
            return 'D'
        return history[-1][opp_index] if len(history) > 0 else 'C'
    elif strategy == 'Malthrin':
        if len(history):
            inp = [(history[i][opp_index] == 'D') for i in range(len(history))]
            if (sum(inp) >= 7) or (len(history) >= (tournament_avg_last_round - 1)):
                return 'D'
        return history[-1][opp_index] if len(history) > 0 else 'C'
    elif strategy == 'Vengeful Cheater':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'D':
            internal_memory[331] = 1
        if (len(history) + 1) < (tournament_avg_last_round - 1):
            return 'D' if internal_memory[331] == 1 else 'C'
        return 'D'
    elif strategy == 'Jesus Petry':
        if not len(history):
            return 'C'
        if len(history) >= (tournament_avg_last_round - 1):
            return 'D'
        if internal_memory[332][2]:
            internal_memory[332][2] -= 1
            if internal_memory[332][2] > 0:
                return 'C'
            return 'C' if 'D' not in (history[-2][opp_index], history[-3][opp_index]) else history[-1][opp_index]
        if (len(history)) < 35:
            if (len(history) + 1) in (22, 35):
                internal_memory[332][2] = 2
                return 'D'
        elif (len(history) + 1) == (internal_memory[332][0] + internal_memory[332][1] - (19 * internal_memory[332][3])):
            A = (internal_memory[332][0] + internal_memory[332][1] - (19 * internal_memory[332][3]))
            internal_memory[332][0] = internal_memory[332][1]
            internal_memory[332][1] = A
            internal_memory[332][2] = 2
            internal_memory[332][3] = 1
            return 'D'
        return history[-1][opp_index]
    elif strategy == 'FAWS':
        if len(history):
            internal_memory[333][1].append(history[-1][opp_index])
            internal_memory[333][0].append(history[-1][self_index])
        turn = len(history) + 1  
        opp_defects = internal_memory[333][1].count('D')
        if opp_defects >= 3:
            return 'D'
        if internal_memory[333][3]:
            return internal_memory[333][3].pop(0)
        if internal_memory[333][4] == 'RULE_10_DEFECTING':
            if len(history) > 0 and internal_memory[333][1][-1] == 'D':
                internal_memory[333][4] = None
                internal_memory[333][3] = ['C', 'C']
            else:
                return 'D'
        if turn >= tournament_avg_last_round - 1: 
            if internal_memory[333][4] == 'USE_RULE_12':
                if opp_defects == 1:
                    return 'C' if turn == 1 else internal_memory[333][1][-1] 
                return 'D'
            return 'D'
        if turn == 1:
            return 'C'
        if turn == 20:
            if internal_memory[333][1][:20].count('D') == 0:
                internal_memory[333][2] = rnd.randint(21, 30)
        if internal_memory[333][2] is not None:
            if 20 < turn < internal_memory[333][2] and internal_memory[333][1][-1] == 'D':
                internal_memory[333][2] = None 
                internal_memory[333][3] = ['D', 'C'] 
                internal_memory[333][4] = 'USE_RULE_12'
                return 'C'
            if turn == internal_memory[333][2]:
                internal_memory[333][2] = None 
                return 'D'
        if len(history) >= 1 and internal_memory[333][0][-1] == 'D' and turn > 21:
            last_turn_idx = len(history) - 1
            if len(history) >= 1 and internal_memory[333][1][-1] == 'D':
                internal_memory[333][4] = 'USE_RULE_12'
                return 'C'
            if len(history) >= 2 and internal_memory[333][1][-1] == 'D':
                return 'C'
            if len(history) >= 3 and internal_memory[333][1][-1] == 'D':
                internal_memory[333][4] = 'USE_RULE_12'
                return 'C'
            if len(history) >= 2 and internal_memory[333][1][-1] == 'C' and internal_memory[333][1][-2] == 'C':
                internal_memory[333][4] = 'RULE_10_DEFECTING'
                return 'D'
        return internal_memory[333][1][-1] if internal_memory[333][1] else 'C'
    elif strategy == 'Second Chance':
        if len(history):
            internal_memory[334][0].append(history[-1][self_index])
            internal_memory[334][1].append(history[-1][opp_index])
        turn = len(history) + 1  
        if turn == 1:
            return 'C'
        if turn >= (tournament_avg_last_round - 2):
            return 'D'
        if internal_memory[334][2]:
            return 'D'
        if len(history) >= 2:
            coop_indices = [
                i for i in range(len(history) - 1) 
                if internal_memory[334][0][i] == 'C'
            ]
            if len(coop_indices) >= 4:
                all_exploited = all(internal_memory[334][1][i] == 'D' for i in coop_indices)
                if all_exploited:
                    internal_memory[334][2] = True
                    return 'D'
        if len(history) >= 2:
            prev_my = internal_memory[334][0][:-1]
            prev_opp = internal_memory[334][1]
            c_count = prev_my.count('C')
            d_count = prev_my.count('D')
            if c_count >= 8 and d_count >= 10:
                followed_by_c_after_my_c = 0
                followed_by_c_after_my_d = 0
                for i in range(len(prev_my)):
                    if prev_my[i] == 'C' and prev_opp[i] == 'C':
                        followed_by_c_after_my_c += 1
                    elif prev_my[i] == 'D' and prev_opp[i] == 'C':
                        followed_by_c_after_my_d += 1
                x = followed_by_c_after_my_c / c_count
                y = followed_by_c_after_my_d / d_count
                if (4 * x) < (6 * y + 1):
                    return 'D'
        opp_defects = internal_memory[334][1].count('D')
        if opp_defects > 0 and opp_defects % 4 == 0:
            return 'C'
        return internal_memory[334][1][-1] if internal_memory[334][1] else 'C'
    elif strategy == 'Simple Identity ChecK':
        turn = len(history) + 1
        if not len(history):
            return 'C'
        if turn <= 57:
            return history[-1][opp_index]
        elif turn == 58:
            return 'D'
        elif turn == 59:
            if history[:57] == ([('C', 'C')] * 57) and history[57] == ('D', 'D'):
                return 'C'
            return 'D'
        return history[-1][opp_index] if history[58][self_index] == 'C' else 'D'
    elif strategy == 'Evil Alliance':
        if len(history) < 5:
            return 'D'
        inp1 = [history[i][opp_index] for i in range(5)]
        inp2 = [history[-i][opp_index] for i in range(1, 5 + 1)] 
        if 'C' in inp1 or ('C' not in inp2 if len(history) > 5 else False):
            return 'D'
        elif len(history) == 5:
            return 'C'
        return history[-1][self_index]
    elif strategy == 'Probe & Punish':
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[335][1] = 11
                internal_memory[335][0] = 2
                return 'D'
        if internal_memory[335][1]:
            internal_memory[335][1] -= 1
            return 'D'
        if internal_memory[335][0]:
            internal_memory[335][0] -= 1
            return 'C'
        return 'C'
    elif strategy == 'Rack Block Shooter':
        """
        0 = Defect, 1 = Cooperate
        """
        handshaking_sequence = ['C', 'D', 'D', 'C']  
        payoff = {
            (1, 1): self_payoff_matrix['CC'],  
            (1, 0): self_payoff_matrix['CD'],  
            (0, 1): self_payoff_matrix['DC'],  
            (0, 0): self_payoff_matrix['DD']   
        }
        if len(history) > 0:
            if len(history) < len(handshaking_sequence):
                if history[-1][opp_index] != handshaking_sequence[len(history)]:
                    internal_memory[336][0] = False
            if history[-1][self_index] == 1:  
                if history[-1][opp_index] == 1:
                    internal_memory[336][1] += 1.0  
                else:
                    internal_memory[336][2] += 1.0   
            else:  
                if history[-1][opp_index] == 1:
                    internal_memory[336][3] += 1.0  
                else:
                    internal_memory[336][4] += 1.0   
        def get_expected_p_cooperate(alpha, beta):
            return alpha / (alpha + beta)
        if len(history) < len(handshaking_sequence):
            return handshaking_sequence[len(history)]
        if internal_memory[336][0]:
            return 'C'  
        p_c_given_c = get_expected_p_cooperate(internal_memory[336][1], internal_memory[336][2])
        p_c_given_d = get_expected_p_cooperate(internal_memory[336][3], internal_memory[336][4])
        eu_cooperate = (p_c_given_c * payoff[(1, 1)]) + ((1 - p_c_given_c) * payoff[(1, 0)])
        eu_defect = (p_c_given_d * payoff[(0, 1)]) + ((1 - p_c_given_d) * payoff[(0, 0)])
        return 'C' if eu_cooperate > eu_defect else 'D'
    elif strategy == 'Control 1':
        if not len(history):
            p = 0.5
        else:
            unziped_memory = list(zip(*history))
            my_history, opponent_history = unziped_memory[self_index], unziped_memory[opp_index]
            last_my_move = my_history[-1]
            matching_indices = [i for i, move in enumerate(my_history[:-1]) if move == last_my_move]
            if not matching_indices:
                p = 0.5
            else:
                cooperations = sum(1 for i in matching_indices if opponent_history[i + 1] == 'C')
                p = cooperations / len(matching_indices)
        Ec = 5 * p - 7
        Ed = 6 * p - 6
        exp_Ec = mth.exp(Ec)
        exp_Ed = mth.exp(Ed)
        prob_cooperate = exp_Ec / (exp_Ec + exp_Ed)
        return 'C' if rnd.random() < prob_cooperate else 'D'
    elif strategy == 'Control 2':
        if not len(history):
            return 'C'
        opponent_history = [history[i][opp_index] for i in range(len(history))]
        if opponent_history[0] == 'D' or opponent_history.count('D') > 2:
            return 'D'
        return 'C'
    elif strategy == 'Control 3':
        turn = len(history) + 1
        if turn == 1:
            return 'D'
        if turn == 2:
            return 'C'
        if history[-1][opp_index] == history[-2][opp_index]:
            return history[-1][opp_index]
        else:
            return flip(history[-1][self_index])
    elif strategy == 'Control 4':
        turn = len(history) + 1
        if turn <= 3:
            return 'C'
        if turn >= (tournament_avg_last_round - 1):
            return 'D'
        opponent_history = [history[i][opp_index] for i in range(len(history))]
        coop_rate = opponent_history.count('C') / len(history)
        return 'C' if coop_rate >= 0.85 else 'D'
    elif strategy == 'Control 5':
        turn = len(history) + 1
        if turn <= 3:
            return 'C'
        if turn == tournament_avg_last_round:  
            return 'D'
        opponent_history = [history[i][opp_index] for i in range(len(history))]
        coop_rate = opponent_history.count('C') / len(history)
        return 'C' if rnd.random() < coop_rate else 'D'
    elif strategy == 'Control 6':
        if not len(history):
            return 'C'
        opponent_history = [history[i][opp_index] for i in range(len(history))]
        if opponent_history[-1] == 'D' and opponent_history.count('D') == 1:
            return 'C'
        return opponent_history[-1]
    elif strategy == 'Control 7':
        turn = len(history) + 1
        if turn == 1:
            return rnd_prob(0.5)
        n = turn // 2  
        target_index = n - 1  
        return history[target_index][opp_index]
    elif strategy == 'Control 8':
        turn = len(history) + 1
        if turn in [20, 40, 60, 80]:
            return 'C'
        if turn > 1:
            unziped_memory = list(zip(*history))
            my_history, opponent_history = unziped_memory[self_index], unziped_memory[opp_index]
            cond1 = len(history) >= 3 and opponent_history[-3:].count('D') >= 2
            cond2 = len(history) >= 2 and my_history[-1] == 'D' and my_history[-2] == 'C'
            cond3 = len(history) >= 10 and 'D' not in my_history[-10:]
            if cond1 or cond2 or cond3:
                return 'D'
        return 'C'
    elif strategy == 'Control 9':
        internal_memory[338][0] = 1  
        internal_memory[338][1] = 0
        internal_memory[338][2] = 0
        internal_memory[338][3] = 0
        turn = len(history) + 1
        if turn > 1:
            unziped_memory = list(zip(*history))
            my_history, opponent_history = unziped_memory[self_index], unziped_memory[opp_index]
            my_last = my_history[-1]
            op_last = opponent_history[-1]
            payoff = {('C', 'C'): self_payoff_matrix['CC'], ('C', 'D'): self_payoff_matrix['CD'], ('D', 'C'): self_payoff_matrix['DC'], ('D', 'D'): self_payoff_matrix['DD']}
            internal_memory[338][3] += payoff[(my_last, op_last)]
            if op_last == 'D':
                internal_memory[338][1] += 1
        if (turn - 1) > 0 and (turn - 1) % 10 == 0 and (turn - 1) != internal_memory[338][2]:
            if 16 <= internal_memory[338][3] <= 34:
                internal_memory[338][0] = 2 if internal_memory[338][0] == 1 else 1
                internal_memory[338][1] = 0  
            internal_memory[338][3] = 0
            internal_memory[338][2] = turn - 1
        if internal_memory[338][0] == 1:  
            if len(history) >= 2:
                if opponent_history[-1] == 'D' and opponent_history[-2] == 'D':
                    return 'D'
            return 'C'
        else:  
            if internal_memory[338][1] >= 1:
                return 'D'
            return 'C'
    elif strategy == 'Control 10':
        turn = len(history) + 1
        AD = (200 / tournament_avg_last_round)
        opponent_history = [history[i][opp_index] for i in range(len(history))]
        if turn in [1, 2]:
            return 'C'
        elif 3 <= turn <= round(29 * AD):
            return opponent_history[-1]  
        elif round(30 * AD) <= turn <= round(84 * AD):
            return 'D' if opponent_history.count('D') > 7 else opponent_history[-1]
        elif turn == round(84 * AD) + 1:
            return 'D'
        elif turn in [round(84 * AD) + 2, round(84 * AD) + 3]:
            return 'D' if opponent_history.count('D') > 7 else 'C'
        elif round(84 * AD) + 4 <= turn <= (tournament_avg_last_round - 2):
            if len(history) >= round(84 * AD) + 3 and (opponent_history[round(84 * AD) + 2] == 'D' or opponent_history.count('D') > 4):
                return 'D'
            return 'C'
        elif (tournament_avg_last_round - 1) <= turn:
            return 'D'
        return 'C'
    elif strategy == 'Control 11':
        turn = len(history) + 1
        opponent_history = [history[i][opp_index] for i in range(len(history))]
        if turn == 1:
            return 'C'
        if turn == 2:
            return opponent_history[-1]
        if turn == tournament_avg_last_round:
            return 'D'
        op_last_d = opponent_history[-1] == 'D'
        my_last_c = history[-1][self_index] == 'C'
        op_prev_d = len(history) >= 2 and opponent_history[-2] == 'D'
        if op_last_d and (my_last_c or op_prev_d):
            return 'D'
        return 'C'
    elif strategy == 'Smarter Tat For Tit':
        if not len(history):
            return 'D'
        if len(history) == 1 and history[0][opp_index] == 'D':
            return 'C'
        return history[-1][opp_index]
    elif strategy == 'Fixed Pattern':
        return 'CDDC'[len(history) % 4]
    elif strategy == 'Signaling Cheater':
        signal = 'DCC'
        if len(history) < len(signal):
            return signal[len(history)]
        inp = [history[i][opp_index] for i in range(len(signal))]
        if inp == signal:
            return 'C'
        return 'D'
    elif strategy == 'Tester Eckhart Arnold Version':
        if not len(history):
            internal_memory[360] = "Test"
            return 'D'
        if internal_memory[360] == "Test":
            internal_memory[360] = "Evaluate"
            return 'D'
        elif internal_memory[360] == "Evaluate":
            if history[-1][opp_index] == 'D':
                internal_memory[360] = "Consolation"
                return 'C'
            else:
                internal_memory[360] = "Dove"
                return 'D'
        elif internal_memory[360] == "Consolation":
            internal_memory[360] = "TitForTat"
            return 'C'
        elif internal_memory[360] == "Dove":
            internal_memory[360] = "Hawk"
            return 'C'
        elif internal_memory[360] == "Hawk":
            internal_memory[360] = "Dove"
            return 'D'
        elif internal_memory[360] == "TitForTat":
            return history[-1][opp_index]
        else:
            raise AssertionError("Tester: state %s is not a valid state!"%internal_memory[360])
    elif strategy == 'Generous Tit For Tat Eckhart Arnold Version 1':
        if not len(history):
            return 'C'
        B = min(5, len(history))
        inp = [history[-i][self_index] for i in range(B, 0, -1)]
        if inp in (['D'] * 5, ['C', 'D', 'C', 'D', 'C']):
            return 'C'
        return history[-1][opp_index]
    elif strategy == 'Generous Tit For Tat Eckhart Arnold Version 2':
        if not len(history):
            return 'C'
        inp = [history[i][opp_index] for i in range(len(history))]
        if inp.count('D') % 5 == 4:
            return 'C'
        return history[-1][opp_index]
    elif strategy == 'Downing Eckhart Arnold Version':
        threshold = 0.9
        if not len(history):
            return 'C'
        if len(history) < 3: return 'C'
        elif len(history) < 6: return 'D'
        elif len(history) == 6: return 'C'
        else:
            i = min(len(history)-1, 20)
            playedD = 0;  punished = 0
            while i > 0:
                if history[-(i+1)][self_index] == 'D':
                    playedD += 1
                    if history[-i][opp_index] == 'D':
                        punished += 1
                i -= 1
            if playedD == 0:
                return history[-1][opp_index]
            else:
                punishFreq = float(punished) / float(playedD)
                if punishFreq > threshold:
                    return 'C'
                else: return 'D'
    elif strategy == 'Tranquilizer Eckhart Arnold Version':
        if not len(history):
            return 'C'
        if internal_memory[361] < 0.5: internal_memory[361] += 1
        if (history[-1][opp_index], (history[-2][opp_index] if len(history) > 1 else 'C')) == ('D', 'D'):
            return 'D'
        return rnd_prob(1 - internal_memory[361])
    elif strategy == 'Parameterized Tit For Tat':
        goodrate = 0.2
        evilrate = 0.05
        if not len(history):
            return rnd_prob(1 - evilrate)
        if history[-1][opp_index] == 'D':
            return rnd_prob(goodrate)
        else:
            return rnd_prob(1 - evilrate)
    elif strategy == 'Tweedledum':
        instruction_string = 'DDHHD'
        instruction = [[instruction_string[4], instruction_string[3]], [instruction_string[2], instruction_string[1]]]
        if not len(history):
            internal_memory[362] = (instruction_string[0] == 'D')
            return 'DC'[internal_memory[362]]
        internal_memory[362] = (instruction[internal_memory[362]][(history[-1][opp_index] == 'C')] == 'D')
        return 'DC'[internal_memory[362]]
    elif strategy == 'Tweedledee':
        instruction_string = 'DDHDD'
        instruction = [[instruction_string[4], instruction_string[3]], [instruction_string[2], instruction_string[1]]]
        if not len(history):
            internal_memory[363] = (instruction_string[0] == 'D')
            return 'DC'[internal_memory[363]]
        internal_memory[363] = (instruction[internal_memory[363]][(history[-1][opp_index] == 'C')] == 'D')
        return 'DC'[internal_memory[363]]
    elif strategy == 'Tweetypie':
        instruction_string = 'HDDHD'
        instruction = [[instruction_string[4], instruction_string[3]], [instruction_string[2], instruction_string[1]]]
        if not len(history):
            internal_memory[364] = (instruction_string[0] == 'D')
            return 'DC'[internal_memory[364]]
        internal_memory[364] = (instruction[internal_memory[364]][(history[-1][opp_index] == 'C')] == 'D')
        return 'DC'[internal_memory[364]]
    elif strategy == 'Simpleton':
        instruction_string = 'HDHDD'
        instruction = [[instruction_string[4], instruction_string[3]], [instruction_string[2], instruction_string[1]]]
        if not len(history):
            internal_memory[365] = (instruction_string[0] == 'D')
            return 'DC'[internal_memory[365]]
        internal_memory[365] = (instruction[internal_memory[365]][(history[-1][opp_index] == 'C')] == 'D')
        return 'DC'[internal_memory[365]]
    elif strategy == 'Humpty 10':
        punish_cycles = 10
        if not len(history):
            internal_memory[366][2] = (internal_memory[366][2] + 1) % len(internal_memory[366][0])
            return internal_memory[366][0][0]
        if internal_memory[366][3] > 0:
            internal_memory[366][3] -= 1
            internal_memory[366][2] = (internal_memory[366][2] + 1) % len(internal_memory[366][0])
            return 'D'
        elif internal_memory[366][3] == 0:
            internal_memory[366][3] -= 1
            move = internal_memory[366][0][internal_memory[366][2]]
            internal_memory[366][2] = (internal_memory[366][2] + 1) % len(internal_memory[366][0])
            return move
        else:
            if history[-1][opp_index] != internal_memory[366][1][internal_memory[366][2]]:
                internal_memory[366][3] = punish_cycles - 1
                internal_memory[366][2] = (internal_memory[366][2] + 1) % len(internal_memory[366][0])
                return 'D'
            else:
                move = internal_memory[366][0][internal_memory[366][2]]
                internal_memory[366][2] = (internal_memory[366][2] + 1) % len(internal_memory[366][0])
                return move
    elif strategy == 'Dumpty 10':
        punish_cycles = 10
        if not len(history):
            internal_memory[367][2] = (internal_memory[367][2] + 1) % len(internal_memory[367][0])
            return internal_memory[367][0][0]
        if internal_memory[367][3] > 0:
            internal_memory[367][3] -= 1
            internal_memory[367][2] = (internal_memory[367][2] + 1) % len(internal_memory[367][0])
            return 'D'
        elif internal_memory[367][3] == 0:
            internal_memory[367][3] -= 1
            move = internal_memory[367][0][internal_memory[367][2]]
            internal_memory[367][2] = (internal_memory[367][2] + 1) % len(internal_memory[367][0])
            return move
        else:
            if history[-1][opp_index] != internal_memory[367][1][internal_memory[367][2]]:
                internal_memory[367][3] = punish_cycles - 1
                internal_memory[367][2] = (internal_memory[367][2] + 1) % len(internal_memory[367][0])
                return 'D'
            else:
                move = internal_memory[367][0][internal_memory[367][2]]
                internal_memory[367][2] = (internal_memory[367][2] + 1) % len(internal_memory[367][0])
                return move
    elif strategy == 'MrX':
        if not len(history):
            return 'D'
        while True:
            if internal_memory[368][0] == "Step 1":
                if history[-1][opp_index] == 'C':  return 'D'
                else:
                    internal_memory[368][0] = "Step 2"
                    internal_memory[368][1] = 5
                    return 'C'
            elif internal_memory[368][0] == "Step 2":
                internal_memory[368][1] -= 1
                if internal_memory[368][1] <= 1:  internal_memory[368][0] = "Step 3"
                return 'C'
            elif internal_memory[368][0] == "Step 3":
                inp = [history[-i][opp_index] for i in range(3, 0, -1)] 
                defections = inp.count('D')
                if defections <= 1: internal_memory[368][0] = "Step 1"
                elif defections == 2: internal_memory[368][0] = "Random"
                else:  internal_memory[368][0] = "TFT"
            elif internal_memory[368][0] == "Random":
                return rnd_prob(0.5)
            else:
                return history[-1][opp_index]
    elif strategy == 'Stefan Pennartz':
        if not len(history):
            return 'C'
        inp = [(history[i][opp_index] == 'C') for i in range(len(history))]
        coop_ratio = sum(inp) / len(history)
        if coop_ratio > 0.5: return history[-1][opp_index]
        else: return 'D'
    elif strategy == 'Friendly Human':
        if not len(history):
            return 'C'

        if (history[-1][opp_index] == 'C') == 1 and (history[-1][self_index] == 'C') == 0: internal_memory[369][3] += 1
        if (history[-1][opp_index] == 'C') == 0 and (history[-1][self_index] == 'C') == 1: internal_memory[369][2] += 1
        if (history[-1][opp_index] == 'C') == 0:
            internal_memory[369][4] += 1
        else:
            internal_memory[369][4] = 0
        internal_memory[369][6] += 1
        if internal_memory[369][6] == 3:
            internal_memory[369][6] = 0
        if len(history) <= 2:
            if (history[-1][opp_index] == 'C') == 0:
                return 'D'
            else:
                return 'C'
        else:
            toVal = ((history[-1][self_index] == 'C')+(history[-2][self_index] == 'C')*2)*10+(history[-1][opp_index] == 'C')+(history[-2][opp_index] == 'C')*2+(history[-3][opp_index] == 'C')*4
            for x in internal_memory[369][1][:]:
                if x == toVal: internal_memory[369][1].remove(x)
            if len(internal_memory[369][1]) < 5:
                return 'D'
            elif internal_memory[369][4] > 4 and len(history) > internal_memory[369][5]:
                internal_memory[369][6] = 1
                internal_memory[369][5] *= 5
                return 'C'
            elif internal_memory[369][6] == 0 and len(history) > 5 and (history[-1][opp_index] == 'C')+(history[-2][opp_index] == 'C')+(history[-3][opp_index] == 'C')+(history[-4][opp_index] == 'C')+(history[-5][opp_index] == 'C') < 3:
                return 'D'
            elif len(history) > 9 and internal_memory[369][2]/internal_memory[369][3] < .90:
                return 'D'
            else:
                returnValue = 0
                for x in internal_memory[369][0][:]:
                    if x == toVal: returnValue = 1
                return 'DC'[returnValue]
    elif strategy == 'Less Grim':
        if not len(history):
            return 'C'
        if len(history) <= 3:  return history[-1][opp_index]
        elif [history[-2][opp_index], history[-1][opp_index]] == ['C','C']:  return 'C'
        elif [history[-3][opp_index], history[-2][opp_index]] == ['C','C'] and history[-2][self_index] == 'D':  return 'C'
        elif history[-1][self_index] == 'D':  return 'D'
        elif history[-1][opp_index] == 'D':  return 'D'
        else:  return 'C'
    elif strategy == 'Generic Downing':
        startRound = 0
        if not len(history):
            return 'C'
        if len(history) - startRound <= 3:  return 'D'
        if len(history) - startRound <= 7:  return 'C'
        pcc = frequency('C', 'C', history, opp_index)
        pcd = frequency('D', 'C', history, opp_index)
        pdc = frequency('C', 'D', history, opp_index)
        if pdc > 0.5:  return 'D'
        if pcd >= pcc:  return 'D'
        return 'C'
    elif strategy == 'Generic Downing Friendly':
        startRound = 0
        if not len(history):
            return 'C'
        if len(history) - startRound <= 3:  return 'C'
        if len(history) - startRound <= 6:  return 'D'
        if len(history) - startRound == 7:  return 'C'
        pcc = frequency('C', 'C', history, opp_index)
        pcd = frequency('D', 'C', history, opp_index)
        pdc = frequency('C', 'D', history, opp_index)
        if pdc > 0.5:  return 'D'
        if pcd >= pcc:  return 'D'
        return 'C'
    elif strategy == 'Smart':
        if not len(history):
            return 'C'
        if internal_memory[370] < 0:
            if history[-1][opp_index] == 'D':
                internal_memory[370] = len(history) - 1
                return 'D'
            else: return 'C'
        else:
            startRound = internal_memory[370]
            if len(history) - startRound <= 3:  return 'D'
            if len(history) - startRound <= 7:  return 'C'
            pcc = frequency('C', 'C', history, opp_index)
            pcd = frequency('D', 'C', history, opp_index)
            pdc = frequency('C', 'D', history, opp_index)
            if pdc > 0.5:  return 'D'
            if pcd >= pcc:  return 'D'
            return 'C'
    elif strategy == 'Absurdistan 1':
        if not len(history):
            return 'C'
        if internal_memory[371][0] == "Phase 1":
            internal_memory[371][1][0] += 1;  internal_memory[371][1][1] += (history[-1][opp_index] == 'C')
            if internal_memory[371][1][0] >= 2 and [history[-2][opp_index], history[-1][opp_index]] == ['D', 'D']:
                internal_memory[371][0] = "Phase 2"
                return 'D'
            return history[-1][opp_index]
        elif internal_memory[371][0] == "Phase 2":
            internal_memory[371][0] = "Phase 3"
            return 'D'
        elif internal_memory[371][0] == "Phase 3":
            internal_memory[371][1][2] += 1;  internal_memory[371][1][3] += (history[-1][opp_index] == 'C')
            if internal_memory[371][1][2] >= 2 and [history[-2][opp_index], history[-1][opp_index]] == ['D', 'D']:
                internal_memory[371][0] = "Phase 4"
                return 'D'
            return history[-1][opp_index]
        elif internal_memory[371][0] == "Phase 4":
            rc1 = internal_memory[371][1][1] / float(internal_memory[371][1][0])
            rc2 = internal_memory[371][1][3] / float(internal_memory[371][1][2])
            internal_memory[371][1][0] = 0;  internal_memory[371][1][1] = 0
            internal_memory[371][1][2] = 0;  internal_memory[371][1][3] = 0
            if rc1 > rc2:
                internal_memory[371][0] = "Phase 5"
                return 'D'
            else:
                internal_memory[371][0] = "Phase 1"
                return 'D'
        elif internal_memory[371][0] == "Phase 5":
            if [history[-2][opp_index], history[-1][opp_index]] == ['C', 'C']:
                internal_memory[371][0] = "Phase 1"
                return 'C'
            else:  return 'D'
        else:  raise AssertionError("Absurdistan1: internal_memory[371][0] = %s not valid"%internal_memory[371][0])
    elif strategy == 'Stabilize':
        if not len(history):
            return 'C'
        if len(history) < 4:
            return history[-1][opp_index]
        pattern = (history[-1][self_index], history[-2][self_index], history[-3][self_index], history[-4][self_index],
                   history[-1][opp_index], history[-2][opp_index], history[-3][opp_index], history[-4][opp_index])
        internal_memory[372].add(pattern)
        if len(internal_memory[372]) > 20:
            return 'D'
        return history[-1][opp_index]
    elif strategy == 'Three Times Out':
        if not len(history):
            return 'C'
        if internal_memory[373]: return 'D'
        if len(history) <= 2:  return 'C'
        inp = [history[-i][opp_index] for i in range(3, 0, -1)] 
        if inp == ['D','D','D']:
            internal_memory[373] = True
            return 'D'
        else: return 'C'
    elif strategy == 'PIN':
        if not len(history):
            return 'D'
        B = min(3, len(history))
        inp = [history[-i][opp_index] for i in range(B, 0, -1)]
        if internal_memory[374]:
            if inp == ['C', 'C', 'C']:
                internal_memory[374] = False
                return 'C'
            else: return 'D'
        else:
            if len(history) >= 3 and inp == ['D', 'D', 'D']:
                internal_memory[374] = True
                return 'D'
            return history[-1][opp_index]
    elif strategy == 'Acquaintance & Trust':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'D':  internal_memory[375] += 1
        if internal_memory[375] > 0:
            B = min(internal_memory[375], len(history))
            inp = [history[-i][opp_index] for i in range(B, 0, -1)]
            if inp == ['C'] * internal_memory[375]:
                internal_memory[375] = 0
                return 'C'
            else:
                return 'D'
        else: return 'C'
    elif strategy == 'Envy':
        if not len(history):
            return 'C'
        unziped_memory = list(zip(*history))
        self_history, opponent_history = unziped_memory[self_index], unziped_memory[opp_index]
        if opponent_history.count('C') < self_history.count('C'):  return 'D'
        else: return 'C'
    elif strategy == 'Soft Grim':
        if not len(history):
            return 'C'
        inp = [history[i][opp_index] for i in range(len(history))]
        if inp.count('C') < len(history) - 3:
            return 'D'
        else:
            return 'C'
    elif strategy == '2 Tits For 2 Tats Eckhart Arnold Version':
        if not len(history):
            return 'C'
        if rnd.random() <= 0.1:
            return rnd_prob(0.5)
        elif len(history) >= 2:
            inp = [history[-i][opp_index] for i in range(1, min(3, len(history)) + 1)]
            if inp[-2:] == ['D', 'D']:
                return 'D'
            elif len(history) >= 3 and inp[-3:-1] == ['D', 'D']:
                return 'D'
            else: return 'C'
        else: return 'C'
    elif strategy == 'Finger Print':
        if not len(history):
            return 'C'
        if len(history) <= 4:
            return ['C', 'D', 'C', 'D', 'C'][len(history)]
        elif len(history) == 5:
            inp = [history[i][opp_index] for i in range(len(history))]
            if inp[0:5] == ['C','C','C','C','C']:
                internal_memory[376] = "B"
            if inp[0:5] == ['C','D','C','D','C'] or inp[0:5] == ['C','C','D','C','D'] or inp == ['D','C','D','C','D']:
                internal_memory[376] = "A"
            else: internal_memory[376] = "C"
            if internal_memory[376] == "A": return 'C'
            else: return 'D'
        else:
            if internal_memory[376] == "A":
                return history[-1][opp_index]
            elif internal_memory[376] == "B":
                if len(history) <= 8:  return 'D'
                elif len(history) == 9:
                    if history[-1][opp_index] == 'D':
                        internal_memory[376] = "D"
                        return 'C'
                    else:
                        internal_memory[376] = "C"
                        return 'D'
            elif internal_memory[376] == "D":
                if len(history) % 2 == 1: return 'D'
                else: return 'C'
            else: return 'D'
    elif strategy == 'Probable Coop':
        return rnd_prob(internal_memory[377])
    elif strategy == 'N Pavlov':
        if len(history) > 0:
            if history[-1] == ('D', 'D'):
                internal_memory[378] = max(0, internal_memory[378] - 2/len(history))
            if history[-1] == ('C', 'C'):
                internal_memory[378] = min(1, internal_memory[378] + 1/len(history))
            if history[-1][self_index] == 'D':
                internal_memory[378] = min(1, internal_memory[378] + 2/len(history))
            if history[-1][self_index] == 'C':
                internal_memory[378] = max(0, internal_memory[378] - 1/len(history))
        return rnd_prob(internal_memory[378])
    elif strategy == 'Memory 1':
        a = rnd.random()
        if not len(history):
            return 'C'
        if history[-1] == ('C', 'C'):
            if a <= internal_memory[379][0]:
                return 'C'
            return 'D'
        if history[-1] == ('D', 'D'):
            if a <= internal_memory[379][1]:
                return 'C'
            return 'D'
        if (history[-1][self_index], history[-1][opp_index]) == ('C', 'D'):
            if a <= internal_memory[379][2]:
                return 'C'
            return 'D'
        if a <= internal_memory[379][3]:
            return 'C'
        return 'D'
    elif strategy == 'Adaptive Hhhpraise Version':
        if len(history) < 3:
            return 'C'
        inp = [history[-i][opp_index] for i in range(3, 0, -1)]
        if inp.count('C') >= 2:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Forgiving':
        if not len(history):
            return 'C'
        if rnd.random() <= 0.5:
            return 'C'
        else:
            return history[-1][opp_index]
    elif strategy == 'Generous Tit For Tat Benjjo Version':
        if not len(history):
            return 'C'
        inp = [(history[-i][self_index] == 'D') for i in range(min(10, len(history)), 0, -1)]
        if sum(inp) >= 10:
            return 'C'
        return history[-1][opp_index]
    elif strategy == 'Graaskamp Benjjo Version':
        if not len(history):
            return 'C'
        if len(history) % 50 == 0:
            return 'D'
        return history[-1][opp_index]
    elif strategy == 'Nydegger Benjjo Version':
        if not len(history):
            return 'C'
        if len(history) >= 5:
            inp = [history[-i][opp_index] for i in range(5, 0, -1)]
            return choice_majority(inp, inp[0])
        else:
            return 'C'
    elif strategy == 'Benjo':
        if len(history) >= 2:
            if (history[-2][opp_index], history[-1][opp_index]) == ('D', 'D'):
                return 'D'
            else:
                inp = [history[i][opp_index] for i in range(len(history))]
                return choice_majority(inp, inp[0])
        else:
            return 'C'
    elif strategy == 'Modal Tit For Tat':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'C':
            return 'C'
        inp = [history[i][opp_index] for i in range(len(history))]
        return choice_majority(inp, inp[0])
    elif strategy == 'Modal Defector':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'C':
            return 'D'
        inp = [history[i][opp_index] for i in range(len(history))]
        return choice_majority(inp, inp[0])
    elif strategy == 'Downing Benjjo Version':
        threshold = 0.7
        if len(history) >= 10:
            inp = [(history[-i][opp_index] == 'C') for i in range(10, 0, -1)]
            c_ratio = sum(inp) / 10
            return 'C' if c_ratio >= threshold else 'D'
        return 'C'
    elif strategy == 'Defect Once':
        return 'D' if not len(history) else 'C'
    elif strategy == 'Cooperate Once':
        return 'C' if not len(history) else 'D'
    elif strategy == 'Forgetful Majority Bot':
        M = 10
        if not len(history):
            return 'C'
        return choice_majority([history[-i][opp_index] for i in range(min(M, len(history)), 0, -1)], 'C')
    elif strategy == 'Prober Abdullah-arif-swe Version':
        if len(history) < 3:
            return 'DCC'[len(history)]
        inp = [history[i][opp_index] for i in range(3)]
        if 'D' in inp:
            return history[-1][opp_index]
        return 'D'
    elif strategy == 'Jester':
        return rnd_prob(0.9)
    elif strategy == 'MSAgent':
        payoff = {
            'CC':(self_payoff_matrix['CC'] / 3) * 0.7,
            'CD':(self_payoff_matrix['CD'] - self_payoff_matrix['DD']) * 0.5,
            'DC':self_payoff_matrix['DC'] / 5,
            'DD':self_payoff_matrix['DD'] * -0.2,
        }
        if len(history) > 0:
            reward1 = payoff[history[-1][self_index] + history[-1][opp_index]]
            reward2 = (history[-1][self_index] == 'C')
            if reward1 >= 0:
                immortal_memory[0]['Trust'] = immortal_memory[0]['Trust'] + ((1 - immortal_memory[0]['Trust']) * reward1)
            else:
                immortal_memory[0]['Trust'] = immortal_memory[0]['Trust'] + (immortal_memory[0]['Trust'] * reward1)
            if reward2 >= 0:
                immortal_memory[0]['Trustworthiness'] = immortal_memory[0]['Trustworthiness'] + ((1 - immortal_memory[0]['Trustworthiness']) * reward2)
            else:
                immortal_memory[0]['Trustworthiness'] = immortal_memory[0]['Trustworthiness'] + (immortal_memory[0]['Trustworthiness'] * reward2)
        if rnd.random() < immortal_memory[0]['Trust']:
            return rnd_prob(immortal_memory[0]['Trustworthiness'])
        return '[OUT]'
    elif strategy == 'Statistical Strategy':
        if len(history) < 2:
            return rnd_prob(0.5)
        confess_total, confess_count = 0.0, 0
        defend_total, defend_count = 0.0, 0
        for i in range(len(history)):
            if history[i][self_index] == 'D':
                confess_total += self_payoff_matrix['D' + history[i][opp_index]]
                confess_count += 1
            elif history[i][self_index] == 'C':
                defend_total += self_payoff_matrix['C' + history[i][opp_index]]
                defend_count += 1
        confess_avg = (confess_total / confess_count) if confess_count > 0 else 0.5
        defend_avg = (defend_total / defend_count) if defend_count > 0 else 0.5
        if confess_avg > defend_avg:
            return 'D'
        elif confess_avg < defend_avg:
            return 'C'
        else:
            return rnd_prob(0.5)
    elif strategy == 'Burn The Bridge Strategy':
        if opp_name in immortal_memory[1]:
            return 'D'
        else:
            A = rnd_prob(0.5)
            if A == 'D':
                immortal_memory[1].append(opp_name)
            return A
    elif strategy == 'Dynocat':
        if not len(history):
            return 'D'
        elif len(history) == 1:
            return history[-1][opp_index]
        elif len(history) < (tournament_avg_last_round - 1):
            return 'D' if (history[-2][opp_index], history[-1][opp_index]) == ('D', 'D') else 'C'
        elif strategy == (tournament_avg_last_round - 1):
            if score[self_index] < score[opp_index]:
                return history[-1][opp_index]
        return 'D'
    elif strategy == 'Darwin':
        if not len(history):
            return 'C'
        if (len(history) - 1) >= len(immortal_memory[2]):
            immortal_memory[2].append(history[-1][opp_index])
        if len(history) > 1:
            if self_payoff_matrix[history[-1][self_index] + history[-1][opp_index]] < self_payoff_matrix['CC']:
                immortal_memory[2][len(history) - 2] = flip(immortal_memory[2][len(history) - 2])
        return immortal_memory[2][len(history) - 1]
    elif strategy == 'WHAgent':
        payoff = {
            'CC':(self_payoff_matrix['CC'] / 3) * 0.7,
            'CD':(self_payoff_matrix['CD'] - self_payoff_matrix['DD']) * 0.5,
            'DC':self_payoff_matrix['DC'] / 5,
            'DD':self_payoff_matrix['DD'] * -0.2,
        }
        if len(history) > 0:
            reward1 = payoff[history[-1][self_index] + history[-1][opp_index]]
            reward2 = (history[-1][self_index] != 'C')
            if reward1 >= 0:
                immortal_memory[5]['Trust'] = immortal_memory[5]['Trust'] + ((1 - immortal_memory[5]['Trust']) * reward1)
            else:
                immortal_memory[5]['Trust'] = immortal_memory[5]['Trust'] + (immortal_memory[5]['Trust'] * reward1)
            if reward2 >= 0:
                immortal_memory[5]['Trustworthiness'] = immortal_memory[5]['Trustworthiness'] + ((1 - immortal_memory[5]['Trustworthiness']) * reward2)
            else:
                immortal_memory[5]['Trustworthiness'] = immortal_memory[5]['Trustworthiness'] + (immortal_memory[5]['Trustworthiness'] * reward2)
        if rnd.random() < immortal_memory[5]['Trust']:
            p_c = coop_prob[1]
            A = 0.5 + abs(p_c - 0.5)
            if rnd.random() < A:
                E = history[-1][opp_index] if len(history) > 0 else 'C'
            else:
                E = flip(history[-1][opp_index]) if len(history) > 0 else 'D'
            if E == 'D':
                return '[OUT]'
        else:
            if opp_name not in ['MSAgent', 'WHAgent', 'RLAgent', 'GossipAgent', 'RLGossipAgent']:
                return '[OUT]'
        return rnd_prob(immortal_memory[5]['Trustworthiness'])
    elif strategy == 'RLAgent':
        payoff = {
            'CC':(self_payoff_matrix['CC'] / 3) * 0.7,
            'CD':(self_payoff_matrix['CD'] - self_payoff_matrix['DD']) * 0.5,
            'DC':self_payoff_matrix['DC'] / 5,
            'DD':self_payoff_matrix['DD'] * -0.2,
        }
        relative_reward = True
        if len(history) > 0:
            gamma = 0.3
            learning_rate = 0.1
            reward1 = payoff[history[-1][self_index] + history[-1][opp_index]]
            reward2 = (history[-1][self_index] != 'C')
            internal_memory[382][0] = reward1 + (gamma * internal_memory[382][0])
            internal_memory[382][1] = 1 + (gamma * internal_memory[382][1])
            internal_memory[382][2] = reward2 + (gamma * internal_memory[382][2])
            internal_memory[382][3] = 1 + (gamma * internal_memory[382][3])
            if relative_reward and internal_memory[382][1] > 0:
                reward1 = reward1 - (internal_memory[382][0] / internal_memory[382][1])
                if reward1 >= 0:
                    immortal_memory[6]['Trust'] = immortal_memory[6]['Trust'] + (learning_rate * ((1 - immortal_memory[6]['Trust']) * reward1))
                else:
                    immortal_memory[6]['Trust'] = immortal_memory[6]['Trust'] + (learning_rate * (immortal_memory[6]['Trust'] * reward1))
            if relative_reward and internal_memory[382][3] > 0:
                reward2 = reward2 - (internal_memory[382][2] / internal_memory[382][3])
                if reward2 >= 0:
                    immortal_memory[6]['Trustworthiness'] = immortal_memory[6]['Trustworthiness'] + (learning_rate * ((1 - immortal_memory[6]['Trustworthiness']) * reward2))
                else:
                    immortal_memory[6]['Trustworthiness'] = immortal_memory[6]['Trustworthiness'] + (learning_rate * (immortal_memory[6]['Trustworthiness'] * reward2))
        if rnd.random() < immortal_memory[6]['Trust']:
            p_c = coop_prob[1]
            A = 0.5 + abs(p_c - 0.5)
            if rnd.random() < A:
                E = history[-1][opp_index] if len(history) > 0 else 'C'
            else:
                E = flip(history[-1][opp_index]) if len(history) > 0 else 'D'
            if E == 'D':
                return '[OUT]'
        else:
            if opp_name not in ['MSAgent', 'WHAgent', 'RLAgent', 'GossipAgent', 'RLGossipAgent']:
                return '[OUT]'
        return rnd_prob(immortal_memory[6]['Trustworthiness'])
    elif strategy == 'GossipAgent':
        payoff = {
            'CC':(self_payoff_matrix['CC'] / 3) * 0.7,
            'CD':(self_payoff_matrix['CD'] - self_payoff_matrix['DD']) * 0.5,
            'DC':self_payoff_matrix['DC'] / 5,
            'DD':self_payoff_matrix['DD'] * -0.2,
        }
        if len(history) > 0:
            reward1 = payoff[history[-1][self_index] + history[-1][opp_index]]
            reward2 = (history[-1][self_index] != 'C')
            if reward1 >= 0:
                immortal_memory[7]['Trust'] = immortal_memory[7]['Trust'] + ((1 - immortal_memory[7]['Trust']) * reward1)
            else:
                immortal_memory[7]['Trust'] = immortal_memory[7]['Trust'] + (immortal_memory[7]['Trust'] * reward1)
            if reward2 >= 0:
                immortal_memory[7]['Trustworthiness'] = immortal_memory[7]['Trustworthiness'] + ((1 - immortal_memory[7]['Trustworthiness']) * reward2)
            else:
                immortal_memory[7]['Trustworthiness'] = immortal_memory[7]['Trustworthiness'] + (immortal_memory[7]['Trustworthiness'] * reward2)
            if reward1 > 0:
                immortal_memory[7]['Memory'][opp_name] = 'GOOD'
            else:
                immortal_memory[7]['Memory'][opp_name] = 'BAD'
        if rnd.random() < immortal_memory[7]['Trust']:
            p_c = coop_prob[1]
            A = 0.5 + abs(p_c - 0.5)
            if rnd.random() < A:
                E = history[-1][opp_index] if len(history) > 0 else 'C'
            else:
                E = flip(history[-1][opp_index]) if len(history) > 0 else 'D'
            if E == 'D':
                return '[OUT]'
        else:
            if opp_name not in ['MSAgent', 'WHAgent', 'RLAgent', 'GossipAgent', 'RLGossipAgent']:
                return '[OUT]'
        if immortal_memory[7]['Memory'].get(opp_name) == 'BAD':
            return '[OUT]'
        return rnd_prob(immortal_memory[7]['Trustworthiness'])
    elif strategy == 'RLGossipAgent':
        payoff = {
            'CC':(self_payoff_matrix['CC'] / 3) * 0.7,
            'CD':(self_payoff_matrix['CD'] - self_payoff_matrix['DD']) * 0.5,
            'DC':self_payoff_matrix['DC'] / 5,
            'DD':self_payoff_matrix['DD'] * -0.2,
        }
        relative_reward = True
        if len(history) > 0:
            gamma = 0.3
            learning_rate = 0.1
            reward1 = payoff[history[-1][self_index] + history[-1][opp_index]]
            reward2 = (history[-1][self_index] != 'C')
            internal_memory[383][0] = reward1 + (gamma * internal_memory[383][0])
            internal_memory[383][1] = 1 + (gamma * internal_memory[383][1])
            internal_memory[383][2] = reward2 + (gamma * internal_memory[383][2])
            internal_memory[383][3] = 1 + (gamma * internal_memory[383][3])
            if relative_reward and internal_memory[383][1] > 0:
                reward1 = reward1 - (internal_memory[383][0] / internal_memory[383][1])
                if reward1 >= 0:
                    immortal_memory[8]['Trust'] = immortal_memory[8]['Trust'] + (learning_rate * ((1 - immortal_memory[8]['Trust']) * reward1))
                else:
                    immortal_memory[8]['Trust'] = immortal_memory[8]['Trust'] + (learning_rate * (immortal_memory[8]['Trust'] * reward1))
            if relative_reward and internal_memory[383][3] > 0:
                reward2 = reward2 - (internal_memory[383][2] / internal_memory[383][3])
                if reward2 >= 0:
                    immortal_memory[8]['Trustworthiness'] = immortal_memory[8]['Trustworthiness'] + (learning_rate * ((1 - immortal_memory[8]['Trustworthiness']) * reward2))
                else:
                    immortal_memory[8]['Trustworthiness'] = immortal_memory[8]['Trustworthiness'] + (learning_rate * (immortal_memory[8]['Trustworthiness'] * reward2))
            if reward1 > 0:
                immortal_memory[8]['Memory'][opp_name] = 'GOOD'
            else:
                immortal_memory[8]['Memory'][opp_name] = 'BAD'
        if rnd.random() < immortal_memory[8]['Trust']:
            p_c = coop_prob[1]
            A = 0.5 + abs(p_c - 0.5)
            if rnd.random() < A:
                E = history[-1][opp_index] if len(history) > 0 else 'C'
            else:
                E = flip(history[-1][opp_index]) if len(history) > 0 else 'D'
            if E == 'D':
                return '[OUT]'
        else:
            if opp_name not in ['MSAgent', 'WHAgent', 'RLAgent', 'GossipAgent', 'RLGossipAgent']:
                return '[OUT]'
        if immortal_memory[8]['Memory'].get(opp_name) == 'BAD':
            return '[OUT]'
        return rnd_prob(immortal_memory[8]['Trustworthiness'])
    elif strategy == 'Second By Robert Adams':
        k40r = (history[-1][self_index] == 'D') if len(history) > 0 else 0
        internal_memory[135]["s"] += 1
        if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 1:
            internal_memory[135]["w"] += 1
            internal_memory[135]["q"] /= 2.0
        if (len(history) + 1) < 3:
            return 'C'
        if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 1:
            w = internal_memory[135]["w"]
            if w > 2 and ((w // 3 == w / 3) or ((w - 1) // 3 == (w - 1) / 3)):
                internal_memory[135]["s"] = 1
                internal_memory[135]["q"] /= 2.0
                return 'D'
            if rnd.random() >= internal_memory[135]["q"]:
                k40r = 0
                internal_memory[135]["q"] /= 2.0
                return 'CD'[k40r]
            else:
                internal_memory[135]["q"] /= 2.0
                return 'D'
        else:
            if internal_memory[135]["s"] == 1 or internal_memory[135]["s"] == 2:
                return 'D'
            w = internal_memory[135]["w"]
            if w > 2 and ((w // 3 == w / 3) or ((w - 1) // 3 == (w - 1) / 3)):
                internal_memory[135]["s"] = 1
                internal_memory[135]["q"] /= 2.0
                return 'D'
            return 'C'
    elif strategy == 'Second By George Hufford':
        internal_memory[381]
        k48r = (history[-1][self_index] == 'D') if len(history) > 0 else 0
        iarray = internal_memory[381]["iarray"]
        ipo2 = internal_memory[381]["ipo2"]
        if (len(history) + 1) == 1:
            internal_memory[381]["kold"] = 0
            internal_memory[381]["k5"] = 0
            internal_memory[381]["klast"] = 0
            for i in range(5):
                iarray[i] = 0
            internal_memory[381]["ichan"] = 1
            internal_memory[381]["ipo1"] = 1
        if (len(history)) < 5:
            idx = ipo2[internal_memory[381]["ipo1"] - 1] - 1
            iarray[idx] = ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
            internal_memory[381]["ipo1"] += ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
            k48r = ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
            return 'CD'[k48r]
        mm = (len(history) % 5) + 1
        k48r = iarray[mm - 1]
        if mm != 1:
            return 'CD'[k48r]
        internal_memory[381]["kold"] = internal_memory[381]["k5"]
        internal_memory[381]["k5"] = score[self_index] - internal_memory[381]["klast"]
        internal_memory[381]["klast"] = score[self_index]
        if internal_memory[381]["kold"] > internal_memory[381]["k5"]:
            internal_memory[381]["ichan"] = -internal_memory[381]["ichan"]
            internal_memory[381]["ipo1"] += internal_memory[381]["ichan"]
        if internal_memory[381]["ipo1"] < 1:
            internal_memory[381]["ipo1"] = 0
        if internal_memory[381]["ipo1"] > 5:
            internal_memory[381]["ipo1"] = 6
        if internal_memory[381]["ipo1"] < 1 or internal_memory[381]["ipo1"] > 5:
            return 'CD'[k48r]
        target_idx1 = ipo2[internal_memory[381]["ipo1"] - 1] - 1
        iarray[target_idx1] += internal_memory[381]["ichan"]
        internal_memory[381]["ipo1"] += internal_memory[381]["ichan"]
        k48r = iarray[mm - 1]
        return 'CD'[k48r]
    elif strategy == 'Second By Nydegger':
        k57r = (history[-1][self_index] == 'D') if len(history) > 0 else 0
        n = 4 * (internal_memory[387]["n"] % 16) + 2 * k57r + ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
        internal_memory[387]["n"] = n
        if (len(history)) < 3:
            k57r = ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
            if (len(history) + 1) == 3 and n == 6:
                k57r = 1
            return 'CD'[k57r]
        coop_states = {
            0, 27, 28, 32,
            40, 41, 42, 43, 44,
            46, 47, 48,
            56, 57,
            59, 60,
            62, 63
        }
        if n in coop_states:
            k57r = 0
        else:
            k57r = 1
        return 'CD'[k57r]
    elif strategy == 'Second By Joss':
        r = rnd.random()
        k69r = (history[-1][self_index] == 'D') if len(history) > 0 else 0
        if (len(history) + 1) == 1:
            return 'C'
        if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 0:
            internal_memory[170]["c"] += 1
        s = internal_memory[170]["s"]
        while True:
            if s == 1:
                if r < 0.1:
                    s = 5
                    k69r = 1
                    break
                if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 0:
                    internal_memory[170]["d"] = 0
                else:
                    internal_memory[170]["d"] += 1
                if internal_memory[170]["d"] > 20:
                    s = 3
                    k69r = 0
                    internal_memory[170]["d"] = 0
                    break
                if internal_memory[170]["c"] < 0.7 * ((len(history) + 1) - 3):
                    s = 2
                    if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 0:
                        internal_memory[170]["d"] = 0
                    else:
                        internal_memory[170]["d"] += 1
                    if internal_memory[170]["d"] > 10:
                        s = 3
                        k69r = 1
                        break
                    k69r = 1
                    break
                k69r = ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
                break
            elif s == 2:
                if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 0:
                    internal_memory[170]["d"] = 0
                else:
                    internal_memory[170]["d"] += 1
                if internal_memory[170]["d"] > 10:
                    s = 3
                    k69r = 1
                    break
                k69r = 1
                break
            elif s == 3:
                if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 0:
                    internal_memory[170]["d"] = 0
                else:
                    internal_memory[170]["d"] += 1
                if internal_memory[170]["d"] > 20:
                    s = 3
                    k69r = 0
                    internal_memory[170]["d"] = 0
                    break
                k69r = ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
                break
            elif s == 4:
                if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 1:
                    internal_memory[170]["f"] += 1
                    if internal_memory[170]["f"] > 3:
                        s = 3
                        k69r = 0
                        internal_memory[170]["d"] = 0
                        break
                s = 1
                k69r = 0
                break
            elif s == 5:
                s = 4
                if r < 0.1:
                    s = 5
                    k69r = 1
                    break
                if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 0:
                    internal_memory[170]["d"] = 0
                else:
                    internal_memory[170]["d"] += 1
                if internal_memory[170]["d"] > 20:
                    s = 3
                    k69r = 0
                    internal_memory[170]["d"] = 0
                    break
                k69r = ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
                break
            break
        internal_memory[170]["s"] = s
        return 'CD'[k69r]
    elif strategy == 'Second By Friedland, But Get Revised By Axelrod':
        if (len(history) + 1) == 1:
            return 'C'
        if internal_memory[390]["jr"] == 1:
            internal_memory[390]["k74dummy"] = 1
            return 'D'
        if (len(history) + 1) > 2:
            if internal_memory[390]["iold"] == 0:
                if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 0:
                    internal_memory[390]["qca"] += 1.0
                internal_memory[390]["qna"] += 1.0
                internal_memory[390]["alpha"] = internal_memory[390]["qca"] / internal_memory[390]["qna"]
                internal_memory[390]["qca"] *= 0.8
                internal_memory[390]["qna"] *= 0.8
            else:
                if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == 0:
                    internal_memory[390]["qcb"] += 1.0
                internal_memory[390]["qnb"] += 1.0
                internal_memory[390]["beta"] = internal_memory[390]["qcb"] / internal_memory[390]["qnb"]
                internal_memory[390]["qcb"] *= 0.8
                internal_memory[390]["qnb"] *= 0.8
        internal_memory[390]["iold"] = internal_memory[390]["k74dummy"]
        if (len(history) + 1) == 37:
            if not (
                internal_memory[390]["js4"] == 0
                or internal_memory[390]["js11"] == 1
                or internal_memory[390]["jt"] <= 10
                or internal_memory[390]["jt"] >= 26
                or internal_memory[390]["jsw"] >= 26
            ):
                internal_memory[390]["jr"] = 1
                internal_memory[390]["k74dummy"] = 1
                return 'D'
        elif (len(history) + 1) > 1:
            if ((history[-1][opp_index] == 'D') if len(history) > 0 else 0) == internal_memory[390]["jl"]:
                internal_memory[390]["jsm"] += 1
                if internal_memory[390]["jsm"] >= 3:
                    internal_memory[390]["js4"] = 1
                if internal_memory[390]["jsm"] >= 11:
                    internal_memory[390]["js11"] = 1
            else:
                internal_memory[390]["jsw"] += 1
                internal_memory[390]["jsm"] = 1
            internal_memory[390]["jt"] += ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
        internal_memory[390]["jl"] = ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
        polc = 6.0 * internal_memory[390]["alpha"] - 8.0 * internal_memory[390]["beta"] - 2.0
        polalt = 4.0 * internal_memory[390]["alpha"] - 5.0 * internal_memory[390]["beta"] - 1.0
        if polc == 0.0:
            k74rxx = 0 if polc >= polalt else 1
        elif polalt >= 0.0:
            k74rxx = 1 - internal_memory[390]["k74dummy"]
        else:
            k74rxx = 1
        internal_memory[390]["k74dummy"] = k74rxx
        return 'CD'[k74rxx]
    elif strategy == 'Second By Feld':
        feld_enemy_action = (history[-1][opp_index] == 'D') if len(history) > 0 else 0
        if internal_memory[391]["ktry"] >= 20:
            idx = internal_memory[391]["jstr"] - 1
            internal_memory[391]["kexp"][idx] = score[self_index] - internal_memory[391]["ki"]
            if internal_memory[391]["jstr"] < 5 and internal_memory[391]["kexp"][idx + 1] > internal_memory[391]["kexp"][idx]:
                internal_memory[391]["jstr"] += 1
            elif internal_memory[391]["jstr"] > 1 and internal_memory[391]["kexp"][idx - 1] > internal_memory[391]["kexp"][idx]:
                internal_memory[391]["jstr"] -= 1
                feld_enemy_action = 0
            internal_memory[391]["ki"] = score[self_index]
            internal_memory[391]["ktry"] = 0
        internal_memory[391]["ktry"] += 1
        jstr = internal_memory[391]["jstr"]
        if jstr == 1:
            return 'C'
        elif jstr == 2:
            if feld_enemy_action == 0:
                return 'C'
            return rnd_prob(1 - 0.75)
        elif jstr == 3:
            return 'CD'[feld_enemy_action]
        elif jstr == 4:
            if feld_enemy_action == 1:
                return 'D'
            return rnd_prob(0.75)
        elif jstr == 5:
            return 'D'
        return 'C'
    elif strategy == 'S2':
        if not len(history):
            return 'D'
        instruction_string = '1:D1|C2,2:D1|D1'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                mode_number, action = item.split(':')
                action_0_str, action_1_str = action.split('|')
                action_0 = [action_0_str[0], action_0_str[1:]]
                action_1 = [action_1_str[0], action_1_str[1:]]
                rule[mode_number] = [action_0, action_1]
            return rule
        instruction = translating(instruction_string)
        rule = instruction[internal_memory[392]][(history[-1][opp_index] == 'D')]
        internal_memory[392] = rule[1]
        return rule[0]
    elif strategy == 'S6':
        if not len(history):
            return 'D'
        instruction_string = '1:C2|D1,2:D1|D1'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                mode_number, action = item.split(':')
                action_0_str, action_1_str = action.split('|')
                action_0 = [action_0_str[0], action_0_str[1:]]
                action_1 = [action_1_str[0], action_1_str[1:]]
                rule[mode_number] = [action_0, action_1]
            return rule
        instruction = translating(instruction_string)
        rule = instruction[internal_memory[393]][(history[-1][opp_index] == 'D')]
        internal_memory[393] = rule[1]
        return rule[0]
    elif strategy == 'S7':
        if not len(history):
            return 'D'
        instruction_string = '1:C2|D1,2:D1|C2'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                mode_number, action = item.split(':')
                action_0_str, action_1_str = action.split('|')
                action_0 = [action_0_str[0], action_0_str[1:]]
                action_1 = [action_1_str[0], action_1_str[1:]]
                rule[mode_number] = [action_0, action_1]
            return rule
        instruction = translating(instruction_string)
        rule = instruction[internal_memory[394]][(history[-1][opp_index] == 'D')]
        internal_memory[394] = rule[1]
        return rule[0]
    elif strategy == 'Grateful / S9':
        if not len(history):
            return 'D'
        instruction_string = '1:C2|D1,2:C2|C2'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                mode_number, action = item.split(':')
                action_0_str, action_1_str = action.split('|')
                action_0 = [action_0_str[0], action_0_str[1:]]
                action_1 = [action_1_str[0], action_1_str[1:]]
                rule[mode_number] = [action_0, action_1]
            return rule
        instruction = translating(instruction_string)
        rule = instruction[internal_memory[395]][(history[-1][opp_index] == 'D')]
        internal_memory[395] = rule[1]
        return rule[0]
    elif strategy == 'S11':
        if not len(history):
            return 'D'
        instruction_string = '1:C2|C2,2:D1|C2'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                mode_number, action = item.split(':')
                action_0_str, action_1_str = action.split('|')
                action_0 = [action_0_str[0], action_0_str[1:]]
                action_1 = [action_1_str[0], action_1_str[1:]]
                rule[mode_number] = [action_0, action_1]
            return rule
        instruction = translating(instruction_string)
        rule = instruction[internal_memory[396]][(history[-1][opp_index] == 'D')]
        internal_memory[396] = rule[1]
        return rule[0]
    elif strategy == 'S12 / Suspicious Forgiver':
        if not len(history):
            return 'D'
        instruction_string = '1:C2|C2,2:C2|D1'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                mode_number, action = item.split(':')
                action_0_str, action_1_str = action.split('|')
                action_0 = [action_0_str[0], action_0_str[1:]]
                action_1 = [action_1_str[0], action_1_str[1:]]
                rule[mode_number] = [action_0, action_1]
            return rule
        instruction = translating(instruction_string)
        rule = instruction[internal_memory[397]][(history[-1][opp_index] == 'D')]
        internal_memory[397] = rule[1]
        return rule[0]
    elif strategy == 'Forgiver Benjamin M. Zagorsky & Johannes G. Reiter & Krishnendu Chatterjee & Martin A. Nowak Version / S14':
        if not len(history):
            return 'C'
        instruction_string = '1:C1|D2,2:C1|C1'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                mode_number, action = item.split(':')
                action_0_str, action_1_str = action.split('|')
                action_0 = [action_0_str[0], action_0_str[1:]]
                action_1 = [action_1_str[0], action_1_str[1:]]
                rule[mode_number] = [action_0, action_1]
            return rule
        instruction = translating(instruction_string)
        rule = instruction[internal_memory[398]][(history[-1][opp_index] == 'D')]
        internal_memory[398] = rule[1]
        return rule[0]
    elif strategy == 'S18':
        if not len(history):
            return 'C'
        instruction_string = '1:D2|C1,2:C1|C1'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                mode_number, action = item.split(':')
                action_0_str, action_1_str = action.split('|')
                action_0 = [action_0_str[0], action_0_str[1:]]
                action_1 = [action_1_str[0], action_1_str[1:]]
                rule[mode_number] = [action_0, action_1]
            return rule
        instruction = translating(instruction_string)
        rule = instruction[internal_memory[399]][(history[-1][opp_index] == 'D')]
        internal_memory[399] = rule[1]
        return rule[0]
    elif strategy == 'S19':
        if not len(history):
            return 'C'
        instruction_string = '1:D2|C1,2:C1|D2'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                mode_number, action = item.split(':')
                action_0_str, action_1_str = action.split('|')
                action_0 = [action_0_str[0], action_0_str[1:]]
                action_1 = [action_1_str[0], action_1_str[1:]]
                rule[mode_number] = [action_0, action_1]
            return rule
        instruction = translating(instruction_string)
        rule = instruction[internal_memory[400]][(history[-1][opp_index] == 'D')]
        internal_memory[400] = rule[1]
        return rule[0]
    elif strategy == 'S23':
        if not len(history):
            return 'C'
        instruction_string = '1:D2|D2,2:C1|D2'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                mode_number, action = item.split(':')
                action_0_str, action_1_str = action.split('|')
                action_0 = [action_0_str[0], action_0_str[1:]]
                action_1 = [action_1_str[0], action_1_str[1:]]
                rule[mode_number] = [action_0, action_1]
            return rule
        instruction = translating(instruction_string)
        rule = instruction[internal_memory[401]][(history[-1][opp_index] == 'D')]
        internal_memory[401] = rule[1]
        return rule[0]
    elif strategy == 'S24':
        if not len(history):
            return 'C'
        instruction_string = '1:D2|D2,2:D2|C1'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                mode_number, action = item.split(':')
                action_0_str, action_1_str = action.split('|')
                action_0 = [action_0_str[0], action_0_str[1:]]
                action_1 = [action_1_str[0], action_1_str[1:]]
                rule[mode_number] = [action_0, action_1]
            return rule
        instruction = translating(instruction_string)
        rule = instruction[internal_memory[402]][(history[-1][opp_index] == 'D')]
        internal_memory[402] = rule[1]
        return rule[0]
    elif strategy == 'Paradoxic Grim / S21':
        if not len(history):
            return 'C'
        instruction_string = '1:D2|C1,2:D2|D2'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                mode_number, action = item.split(':')
                action_0_str, action_1_str = action.split('|')
                action_0 = [action_0_str[0], action_0_str[1:]]
                action_1 = [action_1_str[0], action_1_str[1:]]
                rule[mode_number] = [action_0, action_1]
            return rule
        instruction = translating(instruction_string)
        rule = instruction[internal_memory[403]][(history[-1][opp_index] == 'D')]
        internal_memory[403] = rule[1]
        return rule[0]
    elif strategy == 'Pattern Reader':
        if len(history) < 3:
            if len(history) < 2:
                return 'C'
            return 'D' if (history[0][opp_index] == 'D' and history[1][opp_index] == 'D') else 'C'
        sc = (
            (4 if history[len(history)-1][opp_index] == 'D' else 0) + 
            (2 if history[len(history)-1][self_index] == 'D' else 0) + 
            (2 if history[len(history)-2][opp_index] == 'D' else 0) + 
            (1 if history[len(history)-2][self_index] == 'D' else 0)
            )
        return 'D' if sc >= 3 else 'C'
    elif strategy == 'Forgiving Grudger':
        if not len(history):
            return 'C'
        if history[-1] == ('D', 'D'):
            return 'C' if rnd.random() < 2/7 else 'D'
        return history[-1][opp_index]
    elif strategy == 'Scorched Earth':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'D' or internal_memory[405][0] > 0:
            if ((history[-2][opp_index] if len(history) > 1 else 'C'), history[-1][opp_index]) == ('C', 'D'):
                internal_memory[405][1] += 1
                internal_memory[405][0] = internal_memory[405][1]
            internal_memory[405][0] -= 1
            return 'D'
        else:
            return 'C'
    elif strategy == 'Bookend Cooperator':
        if len(history) < 4:
            return 'C'
        if len(history) == tournament_avg_last_round - 1:
            return 'D'
        if len(history) == tournament_avg_last_round:
            return 'C'
        return history[-1][opp_index]
    elif strategy == 'Patient Cooperator':
        if len(history) < 10:
            return 'C'
        return history[-1][opp_index]
    elif strategy == 'Probe & Exploit':
        if not len(history):
            return 'C'
        if len(history) % 50 == 0:
            return 'D'
        return history[-1][opp_index]
    elif strategy == 'Strategist':
        if not len(history):
            return 'C'
        if history[-1][self_index] == 'C':
            internal_memory[406][0] += 1
            if history[-1][opp_index] == 'C':
                internal_memory[406][1] += 1
        else:
            internal_memory[406][2] += 1
            if history[-1][opp_index] == 'C':
                internal_memory[406][3] += 1
        pc = (internal_memory[406][1] + 1) / (internal_memory[406][0] + 2)
        pd = (internal_memory[406][3] + 1) / (internal_memory[406][2] + 2)
        E_c = (pc * self_payoff_matrix['CC']) + ((1 - pc) * self_payoff_matrix['CD'])
        E_d = (pd * self_payoff_matrix['DC']) + ((1 - pd) * self_payoff_matrix['DD'])
        return 'D' if E_d > E_c else 'C'
    elif strategy == 'Second By GRASR\'s Author':
        turn = len(history) + 1
        r = rnd.random()
        grasr = (history[-1][self_index] == 'D') if len(history) > 0 else 0
        if turn < 1:
            grasr = 0
            return 'CD'[grasr]
        elif turn == 1:
            grasr = 0
            return 'CD'[grasr]
        else:
            if turn < 51:
                grasr = ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
                return 'CD'[grasr]
            elif turn == 51:
                grasr = 1
                return 'CD'[grasr]
            else:
                if turn < 57:
                    if turn > 52:
                        internal_memory[407]["history of move"][turn - 52 - 1] = internal_memory[407]["last move"] + ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
                    grasr = ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
                    if grasr - 1 < 0:
                        internal_memory[407]["last move"] = 2
                    else:
                        internal_memory[407]["last move"] = 4
                    return 'CD'[grasr]
                elif turn == 57:
                    if score[opp_index] <= 135:
                        internal_memory[407]["game status"] = 3
                        grasr = 1
                        return 'CD'[grasr]
                    else:
                        j = internal_memory[407]["history of move"][1]
                        if j == 3:
                            if internal_memory[407]["history of move"][0] == 2:
                                if internal_memory[407]["history of move"][2] == 4:
                                    if internal_memory[407]["history of move"][3] == 2:
                                        internal_memory[407]["game status"] = 4
                                        grasr = 0
                                        if turn >= 118:
                                            internal_memory[407]["game status"] = 2
                                        return 'CD'[grasr]
                            internal_memory[407]["game status"] = 1
                            internal_memory[407]["random counter"] = int(r * 10.0 + 5.0)
                            grasr = 0
                            return 'CD'[grasr]
                        elif j == 4:
                            if internal_memory[407]["history of move"][0] == 3:
                                if internal_memory[407]["history of move"][2] == 3:
                                    internal_memory[407]["game status"] = 2
                                    grasr = 0
                                    return 'CD'[grasr]
                            internal_memory[407]["game status"] = 1
                            internal_memory[407]["random counter"] = int(r * 10.0 + 5.0)
                            grasr = 0
                            return 'CD'[grasr]
                        elif j == 5:
                            if internal_memory[407]["history of move"][0] == 5:
                                if internal_memory[407]["history of move"][2] == 5:
                                    internal_memory[407]["game status"] = 2
                                    grasr = 0
                                    return 'CD'[grasr]
                            internal_memory[407]["game status"] = 1
                            internal_memory[407]["random counter"] = int(r * 10.0 + 5.0)
                            grasr = 0
                            return 'CD'[grasr]
                        else:
                            internal_memory[407]["game status"] = 1
                            internal_memory[407]["random counter"] = int(r * 10.0 + 5.0)
                            grasr = 0
                            return 'CD'[grasr]
                else:
                    if internal_memory[407]["game status"] == 1:
                        if internal_memory[407]["random counter"] <= 0:
                            grasr = 1
                            internal_memory[407]["random counter"] = int(r * 10.0 + 5.0)
                            return 'CD'[grasr]
                        else:
                            grasr = ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
                            internal_memory[407]["random counter"] = internal_memory[407]["random counter"] - 1
                            return 'CD'[grasr]
                    elif internal_memory[407]["game status"] == 2:
                        grasr = ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
                        return 'CD'[grasr]
                    elif internal_memory[407]["game status"] == 3:
                        grasr = 1
                        return 'CD'[grasr]
                    elif internal_memory[407]["game status"] == 4:
                        grasr = 0
                        return 'CD'[grasr]
                    else:
                        grasr = ((history[-1][opp_index] == 'D') if len(history) > 0 else 0)
                        return 'CD'[grasr]
    elif strategy == 'Schwoba Sen Gnitz':
        if not len(history):
            return 'C'
        if not internal_memory[409][0] and history[-1][opp_index] == 'D':
            internal_memory[409][0] = True
        if not internal_memory[409][0]:
            return 'C'
        if len(history) >= 10:
            letzte_10_gegner = [history[-i][opp_index] for i in range(10, 0, -1)]
            if all(zug == 'D' for zug in letzte_10_gegner):
                internal_memory[409][1] = True
        if internal_memory[409][1]:
            return 'D'
        if history[-1][opp_index] == 'D':
            return rnd_prob(0.897)
        return rnd_prob(1 - 0.102)
    elif strategy == 'Chimera':
        PROBE_FSM = "0C8D6C0D3C2D0C3D1D6D2C8D3C1D4D2C5CBDAD7CDD7DFD3D3C8D5D7D3D7CBC8D"
        FSMS = {
            "CCCCC": "0D0CAD4D0CBD5C3D9C7D5CDD6C6DAD8C2DDDEDBD0C9C3D9D7C3CED6CBD1C1D2DCD",
            "CCCCD": "0C0CCD7C7DFDBC3D3DDDCDCC2C5DCC3CBC9D5C2D0C6DADFCED3CEC1C4D4D4CED0D",
            "CCCDC": "CDDD3DFD2D1D8C3C2CFD3C2C7CFCCC8CADCDAD9DCCACAC9DDCEDADCC7CCDCD5DFC",
            "CCCDD": "0C9DDDACAC0CED3C3D4D0D6C0D5C4DECCC5D1CAC5D8D6CFD7C3DBDDDDDDC5CFD6C",
            "CCDCC": "1D4CCC7CCD2C3D3D3D4D6DCD4C5CACBCDD8DEC9D3D2DFDBDFD2C4C3DAC1D5CAD6D",
            "CCDCD": "8CBCDC1C1D5D4C7D1DED6C8DEC8C7C4D6CEDED1D8C6D7D1DCC8C9DFC0C8CCC5D4C",
            "CCDDC": "5CDDEDFCDC6D7D9D9DAD7C1CDD7DFDAD4CED2CBD5D2CFD8D0CCC4DFD3D0C8DCD7D",
            "CCDDD": "6DED7DEC0DAC6C4D7D6DBDAC1C2D4CFC0C6D6D9DCD7D2C8CFC4D2D6CDD6DBD8D8D",
            "CDCCC": "0C0CCD3D6C2DED3D3D0C0DCDAD5CCD3D1C5C5C0C0DBD8CFD5D8CBDDD7DDDDDFD2C",
            "CDCCD": "0C0DCD7D6C2DED3D3D0C0CCDAD5CCD3C1C5C5C0C0DBC8DFD5D8DBDDD7CBDDC4D6C",
            "CDCDC": "1C0C0D3D6C2DEC3D3D0D0CCDAD5C3D3C1C5D5C0C0DBD8CFD5C8CCCDD7DDDDDFD2D",
            "CDCDD": "0C0CCD3C6D2CED3D3D9C0CCCAD5CDC3D1C5C5D0D0D5D8CFD5C8CBCDD7DDCDDFC2C",
            "CDDCC": "0C0DCC3D0C2DED3D3D0D0DCDAD5CCD3D1C5C5C0D0DBD8CFD5D8CBDDD7DDDDDFD4C",
            "CDDCD": "0C0CDDFD6C2DED3D3D0C0DCDDD5DCC3D1D6C5CFD0DBD8DED5D5CBDDD7DDDDDDD2C",
            "CDDDD": "ED1CDCDDCCFD6C6D8DFDDD5D6D5D6D2CCDADCD8C0DCD3CBC5DACDC5D8DDDBC4D2C",
            "DCCCC": "3C9DCD2D3CEC0D8DFDBD5C9C9D4D8C7C5D4CAD7C3C2D7D6DEDCCFD3C3DDCECFDFC",
            "DCCDC": "1C0DCC3D6C2CED3D3D2D0DCCAD5CCC3D1D5D5D0CEDBD8CFD5D8DBDDDBDDD8CFD2C",
            "DCDCD": "0C0DFC6C1D9D9C7D8DBC2CCDDC5CEC9D6C6DAD4C2CCD5CFDADECADDC7D8C3D8CBD",
            "DCDDD": "3D1CED9C9D1D9C8DDC2C8C1CDD5CCD7C0C5CEC5C4CBDBC8C6C2D7D8CACBCAC0D5D",
            "DDCCC": "3D9D9CAD5D5D5D3D3D4DAD1C7DFC3DCDAC6DBD4DBC7C6C1D7C5DDDDC7D5C7D4DFD",
            "DDCCD": "0CBCCD3D6C2C3D0DFC0D0CCDAD5D7D4C4C5D2C0C7DBD8CFD8C8C0D5D7DDDDCFC8C",
            "DDCDC": "0CDCDD3C6C2DEC3D3D5CADCDAD1CCC3D1C5C5D6D0DBD8CFD5D8DBDDD7DDDDDFD2C",
            "DDCDD": "3D1D4DCDBCFD4C1C4C9DFD4DFD7CAD4C1C3DFD5C4DDC4C3C7D0C4D6CFD0DFDCDAD",
            "DDDCD": "CC0DCC3D6C2D8D7C3D0C4D5CFD5DCC3C5C5D5C0D0DBDECFD5D6D2D2D7CDD7C8C8D",
            "DDDDC": "3CFDCDAC2D1DED3DDD0D0DCCAC5CCD7C7C5C0C8D0CBD8CFD5D0CBD6D7DEDDDFD6D",
            "DDDDD": "8D8D1CED6CBCCC0D4CADFD1DDC2C6DDC5D5CDCED2C7D5C6D8DCC5CDDDD2D1D0D2D",
        }
        def decode_fsm(enc):
            trans = {}
            for state in range(16):
                for i, inp in enumerate(['C', 'D']):
                    idx = (state * 2 + i) * 2
                    trans[(state, inp)] = (int(enc[idx], 16), enc[idx + 1])
            return trans
        def _load_fsm(pattern):
            if pattern not in FSMS:
                pattern = "DDDDD"
            if pattern not in internal_memory[410]['cache']:
                data = FSMS[pattern]
                internal_memory[410]['cache'][pattern] = (
                    int(data[0], 16),
                    data[1],
                    decode_fsm(data[2:])
                )
            internal_memory[410]['state'], init_act, internal_memory[410]['trans'] = internal_memory[410]['cache'][pattern]
            return init_act
        CHECK_INTERVAL = 28
        RESET_THRESHOLD = -20
        def _reset_state():
            internal_memory[410]['trans'] = None
            internal_memory[410]['state'] = 0
            internal_memory[410]['probe state'] = 0
            internal_memory[410]['in probe'] = True
            internal_memory[410]['in alld'] = False
        if not len(history):
            internal_memory[410]['probe'] = decode_fsm(PROBE_FSM)
            internal_memory[410]['cache'] = {}
            _reset_state()
        chimera_score = score[self_index] - score[opp_index]
        if internal_memory[410]['in probe']:
            if len(history) == 0:
                return 'C'
            if len(history) < 5:
                internal_memory[410]['probe state'], out = internal_memory[410]['probe'][(internal_memory[410]['probe state'], history[-1][opp_index])]
                return out
            internal_memory[410]['in probe'] = False
            pattern = ''.join([history[m][opp_index] for m in range(5)])
            _load_fsm(pattern)
            key = (internal_memory[410]['state'], history[-1][opp_index])
            if key in internal_memory[410]['trans']:
                internal_memory[410]['state'], out = internal_memory[410]['trans'][key]
                return out
            return 'C'
        if not internal_memory[410]['in alld'] and len(history) >= CHECK_INTERVAL and chimera_score < RESET_THRESHOLD:
            internal_memory[410]['in alld'] = True
        if internal_memory[410]['in alld']:
            return 'D'
        key = (internal_memory[410]['state'], history[-1][opp_index])
        if key in internal_memory[410]['trans']:
            internal_memory[410]['state'], out = internal_memory[410]['trans'][key]
            return out
        return 'C'
    elif strategy == 'Adaptive Cooperator':
        if score[opp_index] > score[self_index] * 2:
            internal_memory[411][1] = True
        elif score[opp_index] > score[self_index]:
            internal_memory[411][0] = True
        if internal_memory[411][1]:
            return 'D'
        elif internal_memory[411][0]:
            if not len(history):
                return 'C'
            else:
                return history[-1][opp_index]
        else:
            return 'C'
    elif strategy == 'Hill Climb':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'D':
            internal_memory[412] += 1 / 100
            if(internal_memory[412] >= 1):
                internal_memory[412] = 0.5
        if history[-1][opp_index] == 'C':
            internal_memory[412] -= 1 / 100
            if(internal_memory[412] <= 0):
                internal_memory[412] = 0.1
        return rnd_prob(internal_memory[412])
    elif strategy == 'Greedy':
        payoff = {
            ('C', 'C'):self_payoff_matrix['CC'],
            ('C', 'D'):self_payoff_matrix['CD'],
            ('D', 'C'):self_payoff_matrix['DC'],
            ('D', 'D'):self_payoff_matrix['DD'],
        }
        UNIFORM = -1
        if not len(history):
            internal_memory[413][1] = UNIFORM
        if len(history) > 0:
            if internal_memory[413][1] == UNIFORM:
                weight = 1 / sum((history[i][self_index] == history[-1][self_index]) for i in range(len(history)))
            else:
                weight = internal_memory[413][1]
            internal_memory[413][0][history[-1][self_index]] += weight * (payoff[(history[-1][self_index], history[-1][opp_index])] - internal_memory[413][0][history[-1][self_index]])
        return max(internal_memory[413][0], key=internal_memory[413][0].get)
    elif strategy == 'Epsilon-Greedy':
        payoff = {
            ('C', 'C'):self_payoff_matrix['CC'],
            ('C', 'D'):self_payoff_matrix['CD'],
            ('D', 'C'):self_payoff_matrix['DC'],
            ('D', 'D'):self_payoff_matrix['DD'],
        }
        UNIFORM = -1
        epsilon = 0.1
        if not len(history):
            internal_memory[414][1] = UNIFORM
        if len(history) > 0:
            if internal_memory[414][1] == UNIFORM:
                weight = 1 / sum((history[i][self_index] == history[-1][self_index]) for i in range(len(history)))
            else:
                weight = internal_memory[414][1]
            internal_memory[414][0][history[-1][self_index]] += weight * (payoff[(history[-1][self_index], history[-1][opp_index])] - internal_memory[414][0][history[-1][self_index]])
        if rnd.random() <= epsilon:
            return rnd_prob(0.5)
        else:
            return max(internal_memory[414][0], key=internal_memory[414][0].get)
    elif strategy == 'AZD / Adaptive Zero Determinant':
        payoff = {
            ('C', 'C'):self_payoff_matrix['CC'],
            ('D', 'C'):self_payoff_matrix['CD'],
            ('C', 'D'):self_payoff_matrix['DC'],
            ('D', 'D'):self_payoff_matrix['DD'],
        }
        if not len(history):
            return 'C'
        phi = 0.125
        s = internal_memory[415]['s']
        l = internal_memory[415]['l']
        if len(history) > 0:
            internal_memory[415]['SCO'][history[-1][self_index]] += payoff[(history[-1][self_index], history[-1][opp_index])]
            d = rnd.randint(0, 9) / 1000
            if internal_memory[415]['SCO']['C'] > internal_memory[415]['SCO']['D']:
                if l + d > self_payoff_matrix['CC']:
                    l = (l + self_payoff_matrix['CC']) / 2
                else:
                    l += d
                clamped_l = max(min(l, self_payoff_matrix['CC']), self_payoff_matrix['DD'])
                s_min = -min((self_payoff_matrix['DC'] - clamped_l) / (clamped_l - self_payoff_matrix['CD']), (clamped_l - self_payoff_matrix['CD']) / (self_payoff_matrix['DC'] - clamped_l))
                if s - d < s_min:
                    s = (s + s_min) / 2
                else:
                    s = s - d
            else:
                if l - d < self_payoff_matrix['DD']:
                    l = (l + self_payoff_matrix['DD']) / 2
                else:
                    l -= d
                clamped_l = max(min(l, self_payoff_matrix['CC']), self_payoff_matrix['DD'])
                s_min = -min((self_payoff_matrix['DC'] - clamped_l) / (clamped_l - self_payoff_matrix['CD']), (clamped_l - self_payoff_matrix['CD']) / (self_payoff_matrix['DC'] - clamped_l))
                if s + d > 1:
                    s = (s + 1) / 2
                else:
                    s += d
            internal_memory[415]['s'] = s
            internal_memory[415]['l'] = l
        phi = max(min(phi, 1), 0)
        s = max(min(s, 1), s_min)
        l = clamped_l
        p1 = 1 - phi * (1 - s) * (self_payoff_matrix['CC'] - l)
        p2 = 1 - phi * (s * (l - self_payoff_matrix['CD']) + (self_payoff_matrix['DC'] - l))
        p3 = phi * ((l - self_payoff_matrix['CD']) + s * (self_payoff_matrix['DC'] - l))
        p4 = phi * (1 - s) * (l - self_payoff_matrix['DD'])
        four_vector = [p1, p2, p3, p4]
        return 'C' if rnd.random() <= four_vector[((2 * (history[-1][self_index] == 'D')) + (history[-1][opp_index] == 'D'))] else 'D'
    elif strategy == 'ZeroResp V5.3':
        zr = internal_memory[416]
        step = len(history) + 1
        eff_len = tournament_avg_round
        remaining = max(0, eff_len - step + 1)
        my_hist = [m[self_index] for m in history]
        opp_hist = [m[opp_index] for m in history]
        zr['pending_reason'] = None
        zr['step_hint'] = 0
        def zr_pay(a, b):
            if a == 'C':
                return self_payoff_matrix['CC'] if b == 'C' else self_payoff_matrix['CD']
            return self_payoff_matrix['DC'] if b == 'C' else self_payoff_matrix['DD']
        def zr_push3(x):
            zr['opp_last3'].append(x)
            if len(zr['opp_last3']) > 4:
                zr['opp_last3'].pop(0)
        def zr_live_coop():
            return 1.0 if zr['opp_len'] == 0 else 1.0 - (zr['opp_defects'] / zr['opp_len'])
        def zr_late_threshold():
            return int(eff_len * 0.75)
        def zr_is_hostile():
            return zr['opp_len'] >= 10 and zr_live_coop() < 0.40
        def zr_is_soft_hostile():
            return zr['opp_len'] >= 10 and zr_live_coop() < 0.65
        def zr_adaptive_epoch():
            if zr['warmth'] < 0.3:
                return 10
            if zr['warmth'] > 0.7:
                return 40
            return 25
        def zr_detect_intent():
            if zr['my_C'] == 0 or zr['opp_len'] < 15:
                return "normal"
            r = zr['opp_defects_after_my_C'] / max(1, zr['my_C'])
            if r > 0.40:
                return "exploitation"
            if r < 0.25 and 0.02 < (zr['opp_defects'] / max(1, zr['opp_len'])) < 0.25:
                return "noise"
            return "normal"
        def zr_classify_tactical():
            if zr['opp_len'] < 15:
                return "adaptive"
            cr = zr_live_coop()
            if cr > 0.95:
                return "cooperator"
            if cr < 0.35:
                return "aggressor"
            if zr['opp_len'] > eff_len * 0.65:
                early = zr['opp_defects'] - zr['late_defects']
                if early <= 3 and zr['late_defects'] >= 1:
                    return "backstabber"
            if zr['my_D'] > 0 and zr['opp_coops_after_my_D'] == 0 and zr['opp_defects'] > 5:
                return "grudger"
            if zr['deadlock'] >= 2 or (zr['opp_len'] > 20 and 0.40 < cr < 0.60):
                return "oscillator"
            return "adaptive"
        def zr_update_target_warmth():
            if zr['opp_len'] < 10:
                return
            ca = zr['opp_coops_after_my_C'] / max(1, zr['my_C'])
            od = zr['opp_defects'] / max(1, zr['opp_len'])
            md = zr['my_D'] / max(1, zr['opp_len'])
            t = 0.5 + 0.3 * (ca - 0.5) - 0.4 * max(0.0, od - 0.02) - 0.2 * max(0.0, md - od)
            if zr['tactical'] == "cooperator":
                t += 0.20
            elif zr['tactical'] == "aggressor":
                t -= 0.30
            elif zr['tactical'] == "oscillator":
                t -= 0.15
            elif zr['tactical'] == "backstabber" and zr['opp_len'] > eff_len * 0.65:
                t -= 0.25
            if zr['forgiveness_exploiter'] >= 2:
                t -= 0.20
            zr['target_warmth'] = max(0.10, min(0.90, t))
        def zr_align_warmth(st):
            if st % 5 == 0 and st <= 25:
                zr['warmth'] += (zr['target_warmth'] - zr['warmth']) * 0.10
            elif st > 25 and st % zr_adaptive_epoch() == 0:
                zr['warmth'] = max(0.10, min(0.90, zr['warmth'] + (zr['target_warmth'] - zr['warmth']) * 0.25))
        def zr_nl_proven():
            return zr['nl_my_flips'] > 0
        def zr_nl_active():
            return zr_nl_proven() and zr['nl_balance'] >= 0
        def zr_is_noisy():
            if zr_nl_active():
                return zr['p_noise_est'] > 0.015 and (zr['cc_pairs'] + zr['nl_my_obs']) > 10
            return zr['p_noise_est'] > 0.015 and zr['cc_pairs'] > 10
        def zr_nl_is_unexplained():
            if zr['bad_standing'] or zr['contrite']:
                return False
            if not my_hist:
                return True
            if zr['nl_provoked_until'] and zr['step_hint'] <= zr['nl_provoked_until']:
                return False
            return my_hist[-1] == 'C'
        def zr_enter_red_line():
            if zr_is_noisy():
                zr['state'] = 'RED_LINE_COOLDOWN'
                zr['is_red_line'] = True
                zr['red_line_cooldown'] = int(12 + 20 * (1 - min(zr['p_noise_est'] * 10, 0.9)))
                zr['warmth'] = 0.15
                zr['queue'].clear()
                zr['debt'] = 0
                return
            zr['state'] = 'RED_LINE'
            zr['is_red_line'] = True
            zr['warmth'] = 0.0
            zr['target_warmth'] = 0.0
            zr['queue'].clear()
            zr['debt'] = 0
            zr['echo_forgive'] = 0
            zr['apology_mode'] = False
        def zr_budget():
            jitter = rnd.randint(0, 2)
            if zr['warmth'] > 0.7:
                jitter += 1
            if zr['warmth'] < 0.4:
                jitter -= 1
            if zr['tactical'] == "aggressor":
                jitter = max(jitter - 2, 0)
            if zr_is_noisy():
                jitter += int(zr['p_noise_est'] * 40)
            return max(2, min(12, 3 + jitter))
        def zr_play(action, reason="normal", intended=None):
            if intended is None:
                intended = action
            zr['intended_prev'] = intended
            zr['last_my'] = action
            zr['last_D_reason'] = reason if action == 'D' else None
            return action
        def zr_close_epoch():
            if zr['is_red_line'] and zr['state'] == 'RED_LINE':
                return
            if zr['epoch_step'] >= zr_adaptive_epoch() and zr['debt'] <= 0 and not zr['queue']:
                zr['epoch_step'] = 0
                zr['systemic'] = 0
                if zr['state'] != 'RED_LINE_COOLDOWN':
                    zr['state'] = 'COOPERATIVE'
        def zr_nl_ladder(st):
            if zr['nl_window'] and (st - zr['nl_window'][-1]) > 12:
                zr['nl_evidence'] = 0
                zr['nl_window'].clear()
            zr['nl_evidence'] += 1
            zr['nl_window'].append(st)
            if zr['nl_pardon_step'] and (st - zr['nl_pardon_step']) <= 6:
                zr['forgiveness_exploiter'] += 1
                zr['nl_pardon_step'] = 0
            if zr['nl_evidence'] == 1:
                zr['nl_pardoned'] += 1
                zr['nl_debt_ledger'] += 1
                zr['pending_reason'] = "nl_deferred"
                return True
            if zr['nl_evidence'] == 2:
                zr['nl_pardoned'] += 1
                zr['nl_debt_ledger'] += 1
                zr['pending_reason'] = "nl_deferred"
                if not any(r.get("kind") == "strike" for r in zr['nl_scheduled']):
                    zr['nl_scheduled'].append({"turn": st + rnd.randint(8, 20), "kind": "strike"})
                forg = zr['opp_coops_after_my_D'] / max(1, zr['my_D'])
                if forg > 0.6 and zr['my_D'] >= 3 and zr['nl_balance'] >= 0 and not zr_is_noisy():
                    zr['nl_harvest_size'] = rnd.randint(3, 4)
                    zr['nl_harvest_left'] = 0
                    zr['nl_harvest_postponed'] = 0
                    zr['nl_harvest_scheduled_at'] = st + rnd.randint(6, 10)
                return True
            if zr['nl_evidence'] >= 3:
                if zr_live_coop() < 0.40 and (not zr_is_noisy() or zr['nl_evidence'] >= 8):
                    zr_enter_red_line()
                    return False
            return None
        def zr_on_defect(st):
            zr['step_hint'] = st
            if zr['bad_standing']:
                zr['bad_standing_turns'] -= 1
                if zr['bad_standing_turns'] <= 0:
                    zr['bad_standing'] = False
                zr['contrite'] = False
                return True
            if zr['last_D_reason'] in ('harvest', 'probe') and zr['tactical'] == 'cooperator':
                zr['last_D_reason'] = None
                return True
            if zr['echo_forgive'] > 0:
                if not (zr_nl_active() and zr_nl_is_unexplained()):
                    zr['echo_forgive'] -= 1
                    return True
            if zr_is_noisy():
                if zr_live_coop() > 0.90 and zr['opp_len'] > 20:
                    return True
                p_forgive = min(0.6, zr['p_noise_est'] * 7 + 0.05 + (zr['warmth'] - 0.5) * 0.2)
                if len(zr['opp_last3']) >= 3 and zr['opp_last3'][-2:] == ['D', 'D']:
                    p_forgive *= 0.3
                if rnd.random() < p_forgive:
                    return True
            maxf = 2 + (1 if zr['warmth'] > 0.60 else 0)
            if zr_is_noisy():
                maxf += 2
            if (
                zr['one_shot_forgives'] < maxf
                and zr['opp_defects'] <= 5
                and st > 5
                and zr['clean_peace'] >= 12
                and zr['debt'] <= 0
                and not zr['queue']
                and zr['state'] == 'COOPERATIVE'
                and zr['late_defects'] == 0
                and not zr_is_soft_hostile()
                and zr['warmth'] > 0.35
            ):
                zr['one_shot_forgives'] += 1
                zr['nl_pardon_step'] = st
                return True
            if zr_nl_active() and zr_nl_is_unexplained():
                handled = zr_nl_ladder(st)
                if handled is not None:
                    return handled
            if zr['intent'] != "noise":
                if zr['debt'] > 0 or zr['queue'] or zr['state'] == 'EQUALIZING':
                    zr['systemic'] += 1
            zr['debt'] += 1
            zr['state'] = 'EQUALIZING'
            thr = 3
            if zr['warmth'] < 0.40 or zr_is_soft_hostile() or zr['late_defects'] > 0:
                thr = 2
            if zr['intent'] == "exploitation":
                thr = 2
            if zr_is_noisy():
                thr = 8
            if zr['systemic'] >= thr:
                zr_enter_red_line()
                return False
            zr['queue'].append(st + 1)
            return False
        def zr_harvest_action(rem):
            if zr_is_noisy():
                if rem > 2:
                    return None
            forg = zr['opp_coops_after_my_D'] / max(1, zr['my_D'])
            low = (zr['opp_defects'] / max(1, zr['opp_len'])) < 0.02
            high = forg > 0.80
            is_victim = high or low
            is_grim = zr['opp_len'] > 50 and zr['opp_defects'] == 0
            if is_grim:
                if rem <= 1:
                    return 'D'
                if zr['probe_fired']:
                    return 'D'
                return None
            if zr['probe_fired'] and rem <= 3:
                return 'D'
            if is_victim and zr['tactical'] in ("cooperator", "adaptive"):
                return 'D'
            return None
        if len(history) > 0:
            last_realized = my_hist[-1]
            if last_realized != zr['intended_prev'] and last_realized == 'D' and zr['intended_prev'] == 'C':
                zr['contrite'] = True
                zr['first_D_was_noise'] = zr['my_D'] <= 1
                zr['bad_standing'] = True
                zr['bad_standing_turns'] = 1
                zr['nl_my_flips'] += 1
            if zr['intended_prev'] == 'C':
                zr['nl_my_obs'] += 1
            zr['nl_balance'] += zr_pay(last_realized, opp_hist[-1]) - zr_pay(opp_hist[-1], last_realized)
            if len(history) >= 2:
                if my_hist[-2] == 'C' and opp_hist[-2] == 'C':
                    zr['cc_pairs'] += 1
                    if opp_hist[-1] == 'D':
                        zr['unprovoked_d'] += 1
                    if zr['cc_pairs'] % 5 == 0 and zr['cc_pairs'] > 10:
                        if zr_nl_active():
                            total = zr['cc_pairs'] + zr['nl_my_obs']
                            est_n = zr['unprovoked_d'] + zr['nl_my_flips']
                        else:
                            total, est_n = zr['cc_pairs'], zr['unprovoked_d']
                        zr['p_noise_est'] = est_n / max(1, total)
        if zr['apology_mode']:
            if len(history) > 0:
                opp_last = opp_hist[-1]
                my_prev = zr['last_my']
                zr['apology_opp_moves'].append(opp_last)
                zr['opp_len'] += 1
                zr_push3(opp_last)
                if opp_last == 'D':
                    zr['opp_defects'] += 1
                    if my_prev == 'C':
                        zr['opp_defects_after_my_C'] += 1
                    if step > zr_late_threshold():
                        zr['late_defects'] += 1
                    zr['clean_peace'] = 0
                else:
                    if my_prev == 'D':
                        zr['opp_coops_after_my_D'] += 1
                    if my_prev == 'C':
                        zr['opp_coops_after_my_C'] += 1
                        zr['clean_peace'] += 1
                    else:
                        zr['clean_peace'] = 0
                if my_prev == 'D':
                    zr['my_D'] += 1
                else:
                    zr['my_C'] += 1
            zr['apology_steps'] -= 1
            if zr['apology_steps'] <= 0:
                if any(m == 'C' for m in zr['apology_opp_moves']):
                    zr['apology_mode'] = False
                    zr['apology_tries'] = 0
                    zr['first_D_was_noise'] = False
                    zr['contrite'] = False
                    zr['apology_opp_moves'] = []
                    zr['debt'] = 0
                    zr['systemic'] = 0
                    zr['queue'].clear()
                    zr['echo_forgive'] = 1
                    zr['state'] = 'COOPERATIVE'
                    zr['epoch_step'] = 0
                    zr['warmth'] = 0.60
                    zr['target_warmth'] = 0.60
                    zr['tactical'] = "adaptive"
                    zr_close_epoch()
                    return zr_play('C', "apology_success", 'C')
                zr['apology_tries'] += 1
                zr['apology_opp_moves'] = []
                if zr['apology_tries'] >= zr['apology_max_tries']:
                    zr_enter_red_line()
                    return zr_play('D', "red_line", 'D')
                zr['apology_mode'] = False
            else:
                zr['epoch_step'] += 1
                zr_close_epoch()
                return zr_play('C', "apology", 'C')
        if (
            not zr['is_red_line']
            and not zr['apology_mode']
            and zr['first_D_was_noise']
            and len(zr['opp_last3']) >= 2
            and zr['opp_last3'][-2:] == ['D', 'D']
            and zr['apology_tries'] < zr['apology_max_tries']
            and zr['opp_len'] < 300
        ):
            zr['apology_mode'] = True
            zr['apology_steps'] = zr_budget()
            zr['apology_opp_moves'] = []
            zr['epoch_step'] += 1
            zr_close_epoch()
            return zr_play('C', "apology", 'C')
        if len(history) > 0 and zr['opp_len'] < len(history):
            opp_last = opp_hist[-1]
            my_prev = zr['last_my']
            if zr['bad_standing']:
                zr['opp_len'] += 1
                zr_push3(opp_last)
                if opp_last == 'D':
                    zr['opp_defects'] += 1
                    if step > zr_late_threshold():
                        zr['late_defects'] += 1
                    zr['clean_peace'] = 0
                else:
                    zr['clean_peace'] += 1 if my_prev == 'C' else 0
                    if my_prev == 'D':
                        zr['opp_coops_after_my_D'] += 1
                if my_prev == 'D':
                    zr['my_D'] += 1
                else:
                    zr['my_C'] += 1
                zr['bad_standing_turns'] -= 1
                if zr['bad_standing_turns'] <= 0:
                    zr['bad_standing'] = False
                zr['contrite'] = False
                if zr['opp_len'] % 5 == 0:
                    zr['intent'] = zr_detect_intent()
                    zr['tactical'] = zr_classify_tactical()
                    zr_update_target_warmth()
                zr_align_warmth(step)
                zr['epoch_step'] += 1
                zr_close_epoch()
                return zr_play('C', "contrite_full", 'C')
            if opp_last == 'D' and zr['contrite']:
                zr['opp_len'] += 1
                zr_push3(opp_last)
                zr['opp_defects'] += 1
                if my_prev == 'C':
                    zr['opp_defects_after_my_C'] += 1
                if step > zr_late_threshold():
                    zr['late_defects'] += 1
                zr['clean_peace'] = 0
                if my_prev == 'D':
                    zr['my_D'] += 1
                else:
                    zr['my_C'] += 1
                zr['contrite'] = False
                zr['consecutive_forgiven'] += 1
                if zr['opp_len'] % 5 == 0:
                    zr['intent'] = zr_detect_intent()
                    zr['tactical'] = zr_classify_tactical()
                    zr_update_target_warmth()
                zr_align_warmth(step)
                zr['epoch_step'] += 1
                zr_close_epoch()
                return zr_play('C', "contrite", 'C')
            zr['opp_len'] += 1
            zr_push3(opp_last)
            if opp_last == 'D':
                zr['opp_defects'] += 1
                if my_prev == 'C':
                    zr['opp_defects_after_my_C'] += 1
                if step > zr_late_threshold():
                    zr['late_defects'] += 1
                if my_prev == 'D':
                    zr['opp_defects_after_my_D'] += 1
                zr_on_defect(step)
                zr['clean_peace'] = 0
            else:
                if my_prev == 'D':
                    zr['opp_coops_after_my_D'] += 1
                if my_prev == 'C':
                    zr['opp_coops_after_my_C'] += 1
                    zr['clean_peace'] += 1
                else:
                    zr['clean_peace'] = 0
            if my_prev == 'D':
                zr['my_D'] += 1
            else:
                zr['my_C'] += 1
        if len(history) > 0:
            if step == 2 and opp_hist[0] == 'D':
                zr['opening_d'] = True
                zr['queue'].clear()
                zr['debt'] = 0
                zr['systemic'] = 0
                zr['state'] = 'COOPERATIVE'
                return zr_play('D', "opening_retort", 'D')
            if step == 3 and zr['opening_d']:
                if opp_hist[1] == 'C':
                    zr['queue'].clear()
                    zr['debt'] = 0
                    zr['systemic'] = 0
                    zr['state'] = 'COOPERATIVE'
                    zr['echo_forgive'] = max(zr['echo_forgive'], 1)
                    return zr_play('C', "opening_resync", 'C')
                zr['queue'].clear()
                zr['debt'] = max(zr['debt'], 1)
                zr['state'] = 'EQUALIZING'
                return zr_play('D', "opening_dd", 'D')
        if zr['nl_harvest_scheduled_at'] != 0 and step >= zr['nl_harvest_scheduled_at']:
            if len(history) > 0 and opp_hist[-1] == 'C':
                zr['nl_harvest_left'] = zr['nl_harvest_size']
                zr['nl_harvest_scheduled_at'] = 0
                zr['nl_provoked_until'] = step + zr['nl_harvest_size'] + 3
            elif zr['nl_harvest_postponed'] < 2:
                zr['nl_harvest_postponed'] += 1
                zr['nl_harvest_scheduled_at'] = step + 3
            else:
                zr['nl_harvest_scheduled_at'] = 0
                zr['nl_harvest_size'] = 0
        if zr['nl_harvest_left'] > 0:
            zr['nl_harvest_left'] -= 1
            return zr_play('D', "nl_harvest_comp", 'D')
 
        zr['nl_scheduled'] = [r for r in zr['nl_scheduled'] if r["turn"] >= step]
        due = next((r for r in zr['nl_scheduled'] if r["turn"] == step), None)
        if due is not None:
            zr['nl_scheduled'].remove(due)
            last3 = zr['opp_last3'][-3:]
            if len(last3) >= 3 and all(mv == 'C' for mv in last3) and zr['opp_defects_after_my_D'] == 0:
                zr['nl_debt_ledger'] += 1
                zr['nl_pardoned'] += 1
                zr['nl_pardon_step'] = step
                if step in zr['queue']:
                    zr['queue'].remove(step)
                    zr['debt'] = max(0, zr['debt'] - 1)
                zr['pending_reason'] = "nl_pardon"
            else:
                zr['queue'].append(step)
                zr['nl_debt_ledger'] = max(0, zr['nl_debt_ledger'] - 1)
 
        if zr['opp_len'] % 5 == 0:
            zr['intent'] = zr_detect_intent()
            zr['tactical'] = zr_classify_tactical()
            zr_update_target_warmth()
        zr_align_warmth(step)
        if zr['intent'] != "noise" and not zr_is_noisy():
            if len(zr['opp_last3']) >= 3 and zr['opp_last3'][-3:] == ['D', 'D', 'D']:
                nl_governs = zr_nl_active() and all(m == 'C' for m in my_hist[-4:-1])
                if (
                    not nl_governs
                    and not zr['contrite']
                    and not zr['bad_standing']
                    and zr['consecutive_forgiven'] < 2
                    and not zr['apology_mode']
                ):
                    zr_enter_red_line()
                    if zr['state'] == 'RED_LINE':
                        return zr_play('D', "red_line", 'D')
        if zr_is_hostile() and not zr['contrite'] and not zr['bad_standing'] and not zr['apology_mode']:
            if not zr_is_noisy():
                zr_enter_red_line()
        if zr['is_red_line']:
            if zr['state'] == 'RED_LINE_COOLDOWN':
                if zr['red_line_cooldown'] <= 0:
                    zr['red_line_cooldown'] = int(15 + 10 * (1 - min(zr['p_noise_est'] * 5, 0.8)))
                    zr['epoch_step'] += 1
                    zr_close_epoch()
                    return zr_play('C', "red_line_probe", 'C')
                zr['red_line_cooldown'] -= 1
                return zr_play('D', "red_line_cooldown", 'D')
            return zr_play('D', "red_line", 'D')
        if zr['deadlock'] >= 3:
            zr['deadlock'] = 0
            zr['queue'].clear()
            zr['debt'] = 0
            zr['echo_forgive'] = 0
            zr['state'] = 'COOPERATIVE'
            zr['systemic'] = max(0, zr['systemic'] - 1)
            zr['epoch_step'] += 1
            zr_close_epoch()
            zr['contrite'] = False
            zr['bad_standing'] = False
            return zr_play('C', "normal", 'C')
        if zr['opp_len'] > 30 and remaining <= zr['harvest_window']:
            harvested = zr_harvest_action(remaining)
            if harvested is not None:
                reason = "harvest" if harvested == 'D' else "harvest_keep_C"
                if zr['probe_fired'] and harvested == 'D':
                    reason = "probe"
                return zr_play(harvested, reason, harvested)
        if step in zr['queue']:
            if zr['contrite'] or zr['bad_standing'] or zr['apology_mode']:
                zr['queue'].remove(step)
                zr['queue'].append(step + 1)
                zr['epoch_step'] += 1
                zr_close_epoch()
                return zr_play('C', "contrite" if zr['contrite'] else "apology", 'C')
            zr['queue'].remove(step)
            zr['debt'] = max(0, zr['debt'] - 1)
            echo_val = 2 if zr_is_noisy() else 1
            zr['echo_forgive'] = max(zr['echo_forgive'], echo_val)
            zr_close_epoch()
            return zr_play('D', "retaliation", 'D')
        zr['epoch_step'] += 1
        zr_close_epoch()
        if zr['contrite']:
            zr['contrite'] = False
        zr['bad_standing'] = False
        reason = zr['pending_reason'] if zr['pending_reason'] is not None else "normal"
        zr['pending_reason'] = None
        return zr_play('C', reason, 'C')
    elif strategy == 'ZeroResp':
        zr0 = internal_memory[417]
        step = len(history) + 1
        base_epoch = 25
        eff_len = tournament_avg_round
        late_threshold = int(eff_len * 0.75)
        def zr0_live_coop():
            return 1.0 if zr0['opp_len'] == 0 else 1.0 - (zr0['opp_defects'] / zr0['opp_len'])
        def zr0_is_hostile():
            return zr0['opp_len'] >= 10 and zr0_live_coop() < 0.4
        def zr0_is_soft_hostile():
            return zr0['opp_len'] >= 10 and zr0_live_coop() < 0.7
        def zr0_enter_red_line():
            zr0['state'] = 'RED_LINE'
            zr0['is_red_line'] = True
            zr0['queue'].clear()
        def zr0_close_epoch():
            if zr0['is_red_line'] or zr0['state'] == 'RED_LINE':
                return
            if zr0['epoch_step'] >= base_epoch and zr0['debt'] <= 0 and not zr0['queue']:
                zr0['epoch_step'] = 0
                zr0['systemic'] = 0
                zr0['state'] = 'COOPERATIVE'
        def zr0_play(action):
            zr0['last_my'] = action
            if action == 'D':
                zr0['my_D'] += 1
            return action
        def zr0_on_defect(st):
            if zr0['debt'] > 0 or zr0['queue'] or zr0['state'] == 'EQUALIZING':
                zr0['systemic'] += 1
            zr0['debt'] += 1
            zr0['state'] = 'EQUALIZING'
            threshold = 3
            if zr0['systemic'] >= 1 or zr0_is_soft_hostile() or zr0['late_defects'] > 0:
                threshold = 2
            if zr0['systemic'] >= threshold:
                zr0_enter_red_line()
                return
            if zr0_is_hostile() or zr0['late_defects'] > 0:
                delay = 1
            else:
                delay = 5 + rnd.randint(1, 10)
            zr0['queue'].append(st + delay)
        if len(history) > 0:
            opp_last = history[-1][opp_index]
            zr0['opp_len'] += 1
            if opp_last == 'D':
                zr0['opp_defects'] += 1
                if step > late_threshold:
                    zr0['late_defects'] += 1
                zr0_on_defect(step)
            else:
                if zr0['last_my'] == 'D':
                    zr0['opp_coops_after_my_D'] += 1
        if zr0_is_hostile():
            zr0_enter_red_line()
        if zr0['late_defects'] >= 2:
            zr0_enter_red_line()
            return zr0_play('D')
        if zr0['is_red_line'] or zr0['state'] == 'RED_LINE':
            return zr0_play('D')
        if step in zr0['queue']:
            zr0['queue'].remove(step)
            zr0['debt'] = max(0, zr0['debt'] - 1)
            zr0_close_epoch()
            return zr0_play('D')
        zr0['epoch_step'] += 1
        zr0_close_epoch()
        return zr0_play('C')
    elif strategy == '4-Duality Optimizer 1':
        waves_greater_than_one = 0
        current_wave_length = 0
        for m in history:
            if m[opp_index] == 'D':
                current_wave_length += 1
            else:
                if current_wave_length > 1:
                    waves_greater_than_one += 1
                current_wave_length = 0
        consecutive_our_defections = 0
        for i in range(len(history) - 1, -1, -1):
            if history[i][self_index] == 'D':
                consecutive_our_defections += 1
            else:
                break
        consecutive_opp_defections = 0
        for i in range(len(history) - 1, -1, -1):
            if history[i][opp_index] == 'D':
                consecutive_opp_defections += 1
            else:
                break
        if consecutive_opp_defections >= 2:
            return 'D'
        if consecutive_our_defections > 0 and consecutive_our_defections < waves_greater_than_one:
            return 'D'
        return 'C'
    elif strategy == 'Bayesian Forgiver':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'C':
            internal_memory[418]['Alpha'] += 1.0
        else:
            internal_memory[418]['Beta'] += 1.0
        total = internal_memory[418]['Alpha'] + internal_memory[418]['Beta']
        mean_cooperation = internal_memory[418]['Alpha'] / total
        variance = (internal_memory[418]['Alpha'] * internal_memory[418]['Beta']) / (total * total * (total + 1))
        uncertainty = variance**0.5
        forgiveness_threshold = (
            internal_memory[418]['Forgiveness Threshold']
            + internal_memory[418]['Uncertainty Factor'] * uncertainty
        )
        if history[-1][opp_index] == 'C':
            return 'C'
        else:
            if mean_cooperation >= forgiveness_threshold:
                return 'C'
            else:
                return 'D'
    elif strategy == 'Geller':
        return clone_opp(rnd_prob(0.5), strategy, history, opp_index, opp_name, internal_memory, immortal_memory, chats, next_chat, score, population)
    elif strategy == 'Geller Cooperator':
        return clone_opp('C', strategy, history, opp_index, opp_name, internal_memory, immortal_memory, chats, next_chat, score, population)
    elif strategy == 'Geller Defector':
        return clone_opp('D', strategy, history, opp_index, opp_name, internal_memory, immortal_memory, chats, next_chat, score, population)
    elif strategy == 'Longterm Tit For Tat':
        if not len(history):
            try:
                internal_memory[419]['noise'] = noise_prob
            except NameError:
                internal_memory[419]['noise'] = 0.0
            return 'C'
        if len(history) == 1:
            return history[-1][opp_index]
        if history[-2][self_index] == 'C':
            internal_memory[419]['n_tft_would_c'] += 1
            if history[-1][opp_index] == 'D':
                internal_memory[419]['n_d_when_tft_would_c'] += 1
            n_expected_ds = internal_memory[419]['n_tft_would_c'] * internal_memory[419]['noise']
            std_expected_ds = np.sqrt(
                internal_memory[419]['noise'] * (1 - internal_memory[419]['noise']) * internal_memory[419]['n_tft_would_c']
            )
            internal_memory[419]['z'] = (internal_memory[419]['n_d_when_tft_would_c'] - n_expected_ds) / max(
                1.0, std_expected_ds
            )
        if internal_memory[419]['n_tft_would_c'] >= 5 and internal_memory[419]['z'] < 2:
            return 'C'
        else:
            return history[-1][opp_index]
    elif strategy == 'ISO':
        if not len(history):
            return 'C'
        mem = internal_memory[420]
        self_history = [m[self_index] for m in history]
        opp_history = [m[opp_index] for m in history]
        def _get_reward(my_strat, opp_strat, init_idx, p_end=1e-2, p_noise=0.0):
            own = my_strat + p_noise * (1.0 - 2.0 * my_strat)
            opp = opp_strat[[0, 2, 1, 3]]
            trans_mat = np.array([
                own * opp,
                own * (1.0 - opp),
                (1.0 - own) * opp,
                (1.0 - own) * (1.0 - opp),
            ]).T
            rewards = np.array([self_payoff_matrix['CC'], self_payoff_matrix['CD'], self_payoff_matrix['DC'], self_payoff_matrix['DD']], dtype=float)
            inv = np.linalg.inv(np.eye(4) - (1.0 - p_end) * trans_mat)
            init_vec = np.zeros(4, dtype=np.float32)
            init_vec[init_idx] = 1.0
            reward = init_vec @ (inv @ rewards - rewards)
            return p_end * float(reward) / (1.0 - p_end)
        def _optimize(opp_model, init_idx):
            opp = np.clip(opp_model, 0.0, 1.0)
            def obj(p):
                return -_get_reward(p, opp, init_idx)
            x0 = np.array([0.5, 0.5, 0.5, 0.5])
            bnds = [(0.0, 1.0)] * 4
            res = minimize(obj, x0, method="L-BFGS-B", bounds=bnds, options={"maxiter": 50})
            return -res.fun, res.x
        def _update_single_ewma(state_ewma, action_val):
            df = mem["discount_factor"]
            state_ewma[0] = df * state_ewma[0] + action_val
            state_ewma[1] = df * state_ewma[1] + 1.0
            return state_ewma[0] / state_ewma[1]
        if len(history) >= 2:
            prev_state = (self_history[-2], opp_history[-2])
            opp_act = 1.0 if opp_history[-1] == 'C' else 0.0
            if prev_state == ('C', 'C'):
                mem["opp_model"][0] = _update_single_ewma(mem["ewma_CC"], opp_act)
            elif prev_state == ('C', 'D'):
                mem["opp_model"][2] = _update_single_ewma(mem["ewma_CD"], opp_act)
            elif prev_state == ('D', 'C'):
                mem["opp_model"][1] = _update_single_ewma(mem["ewma_DC"], opp_act)
            elif prev_state == ('D', 'D'):
                mem["opp_model"][3] = _update_single_ewma(mem["ewma_DD"], opp_act)
        state_map = {('C', 'C'): 0, ('C', 'D'): 1, ('D', 'C'): 2, ('D', 'D'): 3}
        current_state = (self_history[-1], opp_history[-1]) if self_history else ('C', 'C')
        state_idx = state_map.get(current_state, 0)
        expected, my_policy = _optimize(mem["opp_model"], state_idx)
        mem["my_policy"] = my_policy
        return rnd_prob(mem["my_policy"][state_idx])
    elif strategy == 'Cooperate ISO':
        if not len(history):
            return 'C'
        mem = internal_memory[421]
        self_history = [m[self_index] for m in history[:len(history)]]
        opp_history = [m[opp_index] for m in history[:len(history)]]
        last_pair = (self_history[-1], opp_history[-1])
        if last_pair == ('C', 'C'):
            mem["reward_history"].append(self_payoff_matrix['CC'])
        elif last_pair == ('C', 'D'):
            mem["reward_history"].append(self_payoff_matrix['CD'])
        elif last_pair == ('D', 'C'):
            mem["reward_history"].append(self_payoff_matrix['DC'])
        elif last_pair == ('D', 'D'):
            mem["reward_history"].append(self_payoff_matrix['DD'])
        iso_mem = mem["iso_mem"]
        def _get_reward_iso(my_strat, opp_strat, init_idx, p_end=1e-2, p_noise=0.0):
            own = my_strat + p_noise * (1.0 - 2.0 * my_strat)
            opp = opp_strat[[0, 2, 1, 3]]
            trans_mat = np.array([
                own * opp,
                own * (1.0 - opp),
                (1.0 - own) * opp,
                (1.0 - own) * (1.0 - opp),
            ]).T
            rewards = np.array([self_payoff_matrix['CC'], self_payoff_matrix['CD'], self_payoff_matrix['DC'], self_payoff_matrix['DD']], dtype=float)
            inv = np.linalg.inv(np.eye(4) - (1.0 - p_end) * trans_mat)
            init_vec = np.zeros(4, dtype=np.float32)
            init_vec[init_idx] = 1.0
            reward = init_vec @ (inv @ rewards - rewards)
            return p_end * float(reward) / (1.0 - p_end)
        def _optimize_iso(opp_model, init_idx):
            opp = np.clip(opp_model, 0.0, 1.0)
            def obj(p):
                return -_get_reward_iso(p, opp, init_idx)
            x0 = np.array([0.5, 0.5, 0.5, 0.5])
            bnds = [(0.0, 1.0)] * 4
            res = minimize(obj, x0, method="L-BFGS-B", bounds=bnds, options={"maxiter": 50})
            return -res.fun, res.x
        def _update_ewma_iso(state_ewma, val):
            df = iso_mem["discount_factor"]
            state_ewma[0] = df * state_ewma[0] + val
            state_ewma[1] = df * state_ewma[1] + 1.0
            return state_ewma[0] / state_ewma[1]
        if len(history) >= 2:
            prev_state = (self_history[-2], opp_history[-2])
            opp_act = 1.0 if opp_history[-1] == 'C' else 0.0
            if prev_state == ('C', 'C'):
                iso_mem["opp_model"][0] = _update_ewma_iso(iso_mem["ewma_CC"], opp_act)
            elif prev_state == ('C', 'D'):
                iso_mem["opp_model"][2] = _update_ewma_iso(iso_mem["ewma_CD"], opp_act)
            elif prev_state == ('D', 'C'):
                iso_mem["opp_model"][1] = _update_ewma_iso(iso_mem["ewma_DC"], opp_act)
            elif prev_state == ('D', 'D'):
                iso_mem["opp_model"][3] = _update_ewma_iso(iso_mem["ewma_DD"], opp_act)
        state_map = {('C', 'C'): 0, ('C', 'D'): 1, ('D', 'C'): 2, ('D', 'D'): 3}
        state_idx = state_map.get(last_pair, 0)
        expected, my_policy = _optimize_iso(iso_mem["opp_model"], state_idx)
        iso_mem["my_policy"] = my_policy
        if mem["playing_iso"]:
            return rnd_prob(iso_mem["my_policy"][state_idx])
        if len(history) == 1:
            return opp_history[-1]
        if self_history[-2] == 'C':
            mem["n_tft_would_c"] += 1
            if opp_history[-1] == 'D':
                mem["n_d_when_tft_would_c"] += 1
            mem["z"] = (mem["n_d_when_tft_would_c"]) / max(1.0, np.sqrt(mem["n_tft_would_c"]))
        reward_hist = mem["reward_history"]
        expected_gain = expected - np.mean(reward_hist)
        if (
            len(reward_hist) >= 10
            and expected_gain > 2.0 * np.std(reward_hist) / np.sqrt(len(reward_hist))
            and expected_gain > 0.05 * (self_payoff_matrix['CC'] - self_payoff_matrix['DD'])
        ):
            mem["playing_iso"] = True
            return rnd_prob(iso_mem["my_policy"][state_idx])
        if mem["n_tft_would_c"] >= 5 and mem["z"] < 2:
            return 'C'
        else:
            return opp_history[-1]
    elif strategy == 'Meta Hunter Aggressive':
        meta_hunt_team = [
            call_strategy('Defector Hunter', history, opp_index, opp_name, internal_memory[427], None, None, None, None, None),
            call_strategy('Cooperator Hunter', history, opp_index, opp_name, internal_memory[427], None, None, None, None, None),
            call_strategy('Alternator Hunter', history, opp_index, opp_name, internal_memory[427], None, None, None, None, None),
            call_strategy('Random Hunter', history, opp_index, opp_name, internal_memory[427], None, None, None, None, None),
            call_strategy('Math Constant Hunter', history, opp_index, opp_name, internal_memory[427], None, None, None, None, None),
            call_strategy('Cycle Hunter', history, opp_index, opp_name, None, None, None, None, None, None),
            call_strategy('Eventual Cycle Hunter', history, opp_index, opp_name, internal_memory[427], None, None, None, None, None),
        ]
        if len(history) > 100:
            return history[-1][opp_index]
        return 'D' if 'D' in meta_hunt_team else 'C'
    elif strategy == 'Nice Meta Winner Ensemble':
        return transformator(call_strategy('Meta Winner Ensemble', history, opp_index, opp_name, internal_memory[428], None, None, None, None, None), 'NICE', opp_index, history)
    elif strategy == 'Nice Meta Winner':
        return transformator(call_strategy('Meta Winner', history, opp_index, opp_name, internal_memory[429], None, None, None, None, None), 'NICE', opp_index, history)
    elif strategy == 'ZD-X A. J. Stewart & J. B. Plotkin Version / Zero Determinant Extortion A. J. Stewart & J. B. Plotkin Version':
        if not len(history):
            return 'C'
        phi = 1 / 9
        s = 0.5
        l = self_payoff_matrix['DD']
        clamped_l = max(min(l, self_payoff_matrix['CC']), self_payoff_matrix['DD'])
        s_min = -min((self_payoff_matrix['DC'] - clamped_l) / (clamped_l - self_payoff_matrix['CD']), (clamped_l - self_payoff_matrix['CD']) / (self_payoff_matrix['DC'] - clamped_l))
        phi = max(min(phi, 1), 0)
        s = max(min(s, 1), s_min)
        l = clamped_l
        p1 = 1 - phi * (1 - s) * (self_payoff_matrix['CC'] - l)
        p2 = 1 - phi * (s * (l - self_payoff_matrix['CD']) + (self_payoff_matrix['DC'] - l))
        p3 = phi * ((l - self_payoff_matrix['CD']) + s * (self_payoff_matrix['DC'] - l))
        p4 = phi * (1 - s) * (l - self_payoff_matrix['DD'])
        four_vector = [p1, p2, p3, p4]
        return 'C' if rnd.random() <= four_vector[((2 * (history[-1][self_index] == 'D')) + (history[-1][opp_index] == 'D'))] else 'D'
    elif strategy == 'ZD-X Steven Kuhn Version / Zero Determinant Extortion Steven Kuhn Version':
        if not len(history):
            return 'C'
        phi = 1 / 8
        s = 0.5
        l = 1.0
        clamped_l = max(min(l, self_payoff_matrix['CC']), self_payoff_matrix['DD'])
        s_min = -min((self_payoff_matrix['DC'] - clamped_l) / (clamped_l - self_payoff_matrix['CD']), (clamped_l - self_payoff_matrix['CD']) / (self_payoff_matrix['DC'] - clamped_l))
        phi = max(min(phi, 1), 0)
        s = max(min(s, 1), s_min)
        l = clamped_l
        p1 = 1 - phi * (1 - s) * (self_payoff_matrix['CC'] - l)
        p2 = 1 - phi * (s * (l - self_payoff_matrix['CD']) + (self_payoff_matrix['DC'] - l))
        p3 = phi * ((l - self_payoff_matrix['CD']) + s * (self_payoff_matrix['DC'] - l))
        p4 = phi * (1 - s) * (l - self_payoff_matrix['DD'])
        four_vector = [p1, p2, p3, p4]
        return 'C' if rnd.random() <= four_vector[((2 * (history[-1][self_index] == 'D')) + (history[-1][opp_index] == 'D'))] else 'D'
    elif strategy == 'ZD-X / Zero Determinant Extortion':
        if not len(history):
            return 'C'
        phi = 3 / 26
        s = 1 / 3
        l = 1.0
        clamped_l = max(min(l, self_payoff_matrix['CC']), self_payoff_matrix['DD'])
        s_min = -min((self_payoff_matrix['DC'] - clamped_l) / (clamped_l - self_payoff_matrix['CD']), (clamped_l - self_payoff_matrix['CD']) / (self_payoff_matrix['DC'] - clamped_l))
        phi = max(min(phi, 1), 0)
        s = max(min(s, 1), s_min)
        l = clamped_l
        p1 = 1 - phi * (1 - s) * (self_payoff_matrix['CC'] - l)
        p2 = 1 - phi * (s * (l - self_payoff_matrix['CD']) + (self_payoff_matrix['DC'] - l))
        p3 = phi * ((l - self_payoff_matrix['CD']) + s * (self_payoff_matrix['DC'] - l))
        p4 = phi * (1 - s) * (l - self_payoff_matrix['DD'])
        four_vector = [p1, p2, p3, p4]
        return 'C' if rnd.random() <= four_vector[((2 * (history[-1][self_index] == 'D')) + (history[-1][opp_index] == 'D'))] else 'D'
    elif strategy == 'ZD-X Marc Harper Version / Zero Determinant Extortion Marc Harper Version':
        if not len(history):
            return 'C'
        phi = 4 / 17
        s = 0.25
        l = 1.0
        clamped_l = max(min(l, self_payoff_matrix['CC']), self_payoff_matrix['DD'])
        s_min = -min((self_payoff_matrix['DC'] - clamped_l) / (clamped_l - self_payoff_matrix['CD']), (clamped_l - self_payoff_matrix['CD']) / (self_payoff_matrix['DC'] - clamped_l))
        phi = max(min(phi, 1), 0)
        s = max(min(s, 1), s_min)
        l = clamped_l
        p1 = 1 - phi * (1 - s) * (self_payoff_matrix['CC'] - l)
        p2 = 1 - phi * (s * (l - self_payoff_matrix['CD']) + (self_payoff_matrix['DC'] - l))
        p3 = phi * ((l - self_payoff_matrix['CD']) + s * (self_payoff_matrix['DC'] - l))
        p4 = phi * (1 - s) * (l - self_payoff_matrix['DD'])
        four_vector = [p1, p2, p3, p4]
        return 'C' if rnd.random() <= four_vector[((2 * (history[-1][self_index] == 'D')) + (history[-1][opp_index] == 'D'))] else 'D'
    elif strategy == 'Bourgeois':
        if not len(history):
            if self_avg_payoff_matrix > opp_avg_payoff_matrix:
                internal_memory[441] = 'Owner'
        return 'D' if internal_memory[441] == 'Owner' else 'C'
    elif strategy == 'Mixed Strategy':
        if not len(history):
            E_c = self_payoff_matrix['CC'] - self_payoff_matrix['CD']
            E_total = (self_payoff_matrix['CC'] - self_payoff_matrix['CD']) + (self_payoff_matrix['DD'] - self_payoff_matrix['DC'])
            A = (E_c / E_total) if E_total != 0 else 0.5
            internal_memory[435] = 'C' if rnd.random() <= A else 'D'
        return internal_memory[435]
    elif strategy == 'Anti Bourgeois':
        if not len(history):
            if self_avg_payoff_matrix > opp_avg_payoff_matrix:
                internal_memory[442] = 'Owner'
        return 'C' if internal_memory[442] == 'Owner' else 'D'
    elif strategy == 'Primer\'s Strategy':
        if not len(history):
            p_c = population.count('NICE') / len(population)
            p_d = 1 - p_c
            E_c = (p_c * self_payoff_matrix['CC']) + (p_d * self_payoff_matrix['CD'])
            E_d = (p_c * self_payoff_matrix['DC']) + (p_d * self_payoff_matrix['DD'])
            if E_c > E_d:
                internal_memory[436] = 'C'
            elif E_c == E_d:
                internal_memory[436] = rnd_prob(0.5)
            else:
                internal_memory[436] = 'D'
        return internal_memory[436]
    elif strategy == 'Focal Point Strategy / Schelling Point Strategy':
        if not len(history):
            p_c = population.count('NICE') / len(population)
            if p_c > 0.5:
                internal_memory[437] = 'C'
            elif p_c == 0.5:
                internal_memory[437] = rnd_prob(0.5)
            else:
                internal_memory[437] = 'D'
        return internal_memory[437]
    elif strategy == 'Minimax Strategy':
        if not len(history):
            E_c = min(self_payoff_matrix['CC'], self_payoff_matrix['CD'])
            E_d = min(self_payoff_matrix['DC'], self_payoff_matrix['DD'])
            if E_c > E_d:
                internal_memory[438] = 'C'
            elif E_c == E_d:
                internal_memory[438] = rnd_prob(0.5)
            else:
                internal_memory[438] = 'D'
        return internal_memory[438]
    elif strategy == 'Payoff-Dominance Strategy':
        if not len(history):
            if self_payoff_matrix['CC'] > self_payoff_matrix['DD']:
                internal_memory[439] = 'C'
            elif self_payoff_matrix['CC'] == self_payoff_matrix['DD']:
                internal_memory[439] = rnd_prob(0.5)
            else:
                internal_memory[439] = 'D'
        return internal_memory[439]
    elif strategy == 'Risk-Dominance Strategy':
        if not len(history):
            E_c = self_payoff_matrix['CC'] + self_payoff_matrix['CD']
            E_d = self_payoff_matrix['DC'] + self_payoff_matrix['DD']
            if E_c > E_d:
                internal_memory[440] = 'C'
            elif E_c == E_d:
                internal_memory[440] = rnd_prob(0.5)
            else:
                internal_memory[440] = 'D'
        return internal_memory[440]
    elif strategy == 'Local Copycat':
        if not len(history):
            return rnd_prob(0.5)
        if self_payoff_matrix[history[-1][self_index] + history[-1][opp_index]] >= self_payoff_matrix[history[-1][opp_index] + history[-1][self_index]]:
            return history[-1][self_index]
        return history[-1][opp_index]
    elif strategy == 'Alternating Strategy':
        BoS_index = (opp_payoff_matrix['DD'] > opp_payoff_matrix['CC'])
        odd = 'D'
        even = 'C'
        if len(history) > 0:
            last_round_odd = (len(history) % 2 != 0)
            last_round_target = odd if last_round_odd else even
            p1_betray = (history[-1][BoS_index] != last_round_target)
            p2_betray = (history[-1][1 - BoS_index] != last_round_target)
            if p1_betray or p2_betray:
                return 'D' if self_index == BoS_index else 'C'
        return odd if len(history) % 2 == 0 else even
    elif strategy == 'Strategic Commitment':
        BoS_index = (opp_payoff_matrix['DD'] > opp_payoff_matrix['CC'])
        if self_index == BoS_index:
            return 'D'
        if opp_payoff_matrix['DD'] > opp_payoff_matrix['DC']:
            return 'D'
        return 'C'
    elif strategy == 'Backward Induction':
        BoS_index = (opp_payoff_matrix['DD'] > opp_payoff_matrix['CC'])
        A = [self_payoff_matrix, opp_payoff_matrix] if self_index == BoS_index else [opp_payoff_matrix, self_payoff_matrix]
        response_p2_if_p1_D = 'D' if A[1 - BoS_index]['DD'] > A[1 - BoS_index]['DC'] else 'C'
        payoff_p1_on_branch_A = A[BoS_index]['D' + response_p2_if_p1_D]
        response_p2_if_p1_C = 'C' if A[1 - BoS_index]['CC'] > A[1 - BoS_index]['CD'] else 'D'
        payoff_p1_on_branch_B = A[BoS_index]['C' + response_p2_if_p1_C]
        if payoff_p1_on_branch_A > payoff_p1_on_branch_B:
            husband_choice = 'D'
            wife_choice = response_p2_if_p1_D
        else:
            husband_choice = response_p2_if_p1_C
            wife_choice = 'C'
        return husband_choice if self_index == BoS_index else wife_choice
    elif strategy == 'Fictitious Play':
        if not len(history):
            return rnd_prob(0.5)
        E_c = sum(self_payoff_matrix['C' + i[opp_index]] for i in history)
        E_d = sum(self_payoff_matrix['D' + i[opp_index]] for i in history)
        if E_d == E_c:
            return rnd_prob(0.5)
        return 'CD'[(E_d > E_c)]

    # Strategies That From This Tournament
    elif strategy == 'Pavlov Tester':
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                if internal_memory[3][1] < 2:
                    internal_memory[3][1] += 1
            else:
                if internal_memory[3][1] > 0:
                    internal_memory[3][1] -= 1
        if internal_memory[3][0] != 3 and internal_memory[3][1] < 2:
            if internal_memory[3][0] < 2:
                internal_memory[3][0] += 1
                return 'D'
            else:
                internal_memory[3][0] = 0
                return 'C'
        else:
            if internal_memory[3][0] != 3:
                internal_memory[3][0] = 3
                return 'C'
            else:
                if not len(history):
                    return 'C'
                return history[-1][opp_index]
    elif strategy == 'Clement Coldridge':
        if len(history) > 0:
            if history[-1][opp_index] == 'D' or internal_memory[4] == 1:
                if rnd.random() < 0.2:
                    internal_memory[4] = 0
                    return 'C'
                else:
                    internal_memory[4] = 1
                    return 'D'
        return 'C'
    elif strategy == 'Gaslighter':
        SH = (history[-1][opp_index] == 'C') if len(history) > 0 else True
        if not SH:
            if (len(history) + 1) > 2 and history[-1-1][opp_index] == 'D':
                return 'D'
            elif rnd.random() < (1 - (internal_memory[6][1] / (len(history) + 1))):
                return 'D'
        else:
            internal_memory[6][1] += 1
        rand_est = 1 - (internal_memory[6][1] / (len(history) + 1))
        A = ((1 - internal_memory[6][0]) * SH) + (internal_memory[6][0] * (internal_memory[6][1] / (len(history) + 1)))
        A = A * (1 - 0.3 * rand_est)
        internal_memory[6][0] = A
        if rnd.random() > (A - 0.05):
            return 'D'
        else:
            return 'C'
    elif strategy == 'Springter':
        if (len(history) + 1) % 10 == 0 or internal_memory[8][1] == 1:
            if internal_memory[8][1] == 0: 
                internal_memory[8][1] = 1
            if history[-1][opp_index] == 'D': 
                internal_memory[8][0] += 1
            if internal_memory[8][0] > 0:
                internal_memory[8][0] -= 1
                return 'D'
            else:
                internal_memory[8][1] = 0
                return 'C'
        elif (history[-1][opp_index] == 'D') if len(history) > 0 else False:
            internal_memory[8][0] += 1
            return 'C'
        else:
            return 'C'
    elif strategy == 'William':
        G = rnd.random()
        if ((history[-1][opp_index] == 'D') if len(history) > 0 else False) and G > 0.2:
            return 'D'
        SH = (history[-1][opp_index] == 'C') if len(history) > 0 else True
        if SH:
            internal_memory[9][0] += 1
        elif G <= 0.2:
            internal_memory[9][1] += 1
        elif (history[-1-1][opp_index] == 'D') if (len(history) + 1) > 2 else False:
            internal_memory[9][1] -= 1
        internal_memory[9][1] = max(internal_memory[9][1], 0)
        A = min(((SH * (internal_memory[9][0] / (len(history) + 1))) + (internal_memory[9][1] / (len(history) + 1))),1)
        if rnd.random() > A:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Random\'s Exploiter 1':
        if not len(history): return 'C'
        if history[-1][opp_index] == 'C':
            internal_memory[11][0] += 1
        else:
            internal_memory[11][1] += 1
        E1 = 20
        E2 = E1/2
        H = ((((internal_memory[11][0] - E2)*(internal_memory[11][0] - E2)))/E2)+((((internal_memory[11][1] - E2)*(internal_memory[11][1] - E2)))/E2)
        if ((len(history) + 1) % E1 == 0 and H < 3.84) or internal_memory[11][2] == 1:
            internal_memory[11][2] = 1
            return 'D'
        else:
            return history[-1][opp_index]
    elif strategy == 'Stoic Mirror':
        if (len(history) + 1) < 4:
            return 'C'
        else:
            if internal_memory[12][1] > 0:
                if internal_memory[12][1] == 5:
                    internal_memory[12][0] = 0
                    internal_memory[12][1] = 0
                    internal_memory[12][2] = 1
                    return 'C'
                internal_memory[12][1] += 1
                return 'D'
            elif internal_memory[12][2] == 1:
                if history[-1][opp_index] == 'D':
                    internal_memory[12][2] = 2
                    return 'D'
                else:
                    internal_memory[12][2] = 0
                    return 'C'
            elif internal_memory[12][2] == 2:
                return 'D'
            else:
                if history[-1][opp_index] == 'D':
                    internal_memory[12][0] += 1
                ratio = internal_memory[12][0] / (len(history) + 1)
                if ratio > 0.3:
                    internal_memory[12][1] = 1
                    return 'D'
                else:
                    return 'C'
    elif strategy == 'Adaptive Tit For Tat Attariq Version':
        if (history[-1][opp_index] == 'C') if len(history) > 0 else True:
            internal_memory[13][1] += 1
        elif rnd.random() < (0.25 * (internal_memory[13][1] / (len(history) + 1))):
            return 'C'
        expectation = internal_memory[13][1] / (len(history) + 1)
        reality = (history[-1][opp_index] == 'C') if len(history) > 0 else True
        A = ((1-internal_memory[13][0])*reality)+(internal_memory[13][0]*expectation)
        internal_memory[13][0] = A
        if A > (1 - expectation):
            return 'C'
        else:
            return 'D'
    elif strategy == 'Entropy Sentinel':
        opp = history[-1][opp_index] if len(history) > 0 else 'C'
        tm = internal_memory[15]
        if opp == 'C':
            tm[0] += 1
        else:
            tm[2] += 1  
        tm[1] += 1
        share_ratio = tm[0] / max(1, tm[1])
        if opp == 'C':
            tm[2] = 0
        if tm[1] < 5:
            if not len(history):
                return 'C'
            return history[-1][opp_index]  
        if share_ratio > 0.6:
            tm[4] += 0.1
        elif share_ratio < 0.4:
            tm[4] -= 0.1
        tm[4] = max(0, min(1, tm[4]))
        if tm[4] > 0.7:
            tm[3] = 1
        elif tm[4] < 0.3:
            tm[3] = 2
        else:
            tm[3] = 0
        mode = tm[3]
        if mode == 1:
            return 'D' if rnd.random() < 0.7 else 'C'
        elif mode == 2:
            return 'C' if tm[2] < 2 else 'D'
        else:
            return history[-1][opp_index] if rnd.random() < 0.8 else rnd.choice(['C','D'])
    elif strategy == 'Learner':
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[17] += 0.125
            else:
                internal_memory[17] -= 0.125
        internal_memory[17] = max(min(internal_memory[17],1),0)
        if rnd.random() < internal_memory[17]:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Paranoid':
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[18] = 1
        if rnd.random() < 0.25 or internal_memory[18] == 1:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Liar Person':
        if (history[-1][opp_index] == 'C') if len(history) > 0 else True:
            internal_memory[19] += 1
        ET = ((len(history) + 1) % 5 == 0)
        A = min(((internal_memory[19] / (len(history) + 1)) * ET) + (1 - (internal_memory[19] / (len(history) + 1))),1)
        if A > 0.5:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Keyman':
        key_len = len(internal_memory[20][0])
        if len(history) > 0:
            if history[-1][opp_index] == 'C':
                internal_memory[20][2] += 1
        if (len(history)) < key_len:
            if len(history) > 0:
                internal_memory[20][1][(len(history) + 1)-1] = (history[-1][opp_index] == 'D')
            key = internal_memory[20][0][(len(history) + 1)-1]
            if key == 1:
                return 'D'
            else:
                return 'C'
        if (len(history) + 1) == key_len:
            internal_memory[20][1][(len(history) + 1)-1] = (history[-1][opp_index] == 'D')
        if internal_memory[20][1] == internal_memory[20][0]:
            return 'C'
        else:
            if (len(history) + 1) > (key_len+1):
                ET = ((len(history) + 1) % 5 == 0)
                A = min(((internal_memory[20][2] / (len(history) + 1)) * ET) + (1 - (internal_memory[20][2] / (len(history) + 1))),1)
                if A > 0.5:
                    return 'D'
                else:
                    return 'C'
            else:
                return 'C'
    elif strategy == 'GT-E / Game Theory Explorer':
        if (history[-1][opp_index] == 'C') if len(history) > 0 else True:
            if (history[-1][self_index] == 'C') if len(history) > 0 else True:
                C = 0
                S = 2
                learn = 2 * (self_payoff_matrix['CC'] / 3)
            else:
                C = 1
                S = 3
                learn = 4 * (self_payoff_matrix['DC'] / 5)
        else:
            if history[-1][self_index] == 'C':
                C = 2
                S = 0
                learn = -3 * (self_payoff_matrix['DD'] - self_payoff_matrix['CD'])
            else:
                C = 3
                S = 1
                learn = self_payoff_matrix['DD']
        if (history[-1][opp_index] == 'C')  if len(history) > 0 else True:
            internal_memory[21][4] += 1
        p_c = internal_memory[21][4] / (len(history) + 1)
        CHO = (history[-1][self_index] == 'D') if len(history) > 0 else False
        internal_memory[21][C][CHO] = max(min(
            internal_memory[21][C][CHO] + learn,
            30
        ), 0)
        internal_memory[21][S][CHO] = max(min(
            internal_memory[21][S][CHO] + learn,
            30
        ), 0)
        opposite = 1 - CHO
        internal_memory[21][C][opposite] = max(min(
            internal_memory[21][C][opposite] - (learn * 0.3),
            30
        ), 0)
        internal_memory[21][S][opposite] = max(min(
            internal_memory[21][S][opposite] - (learn * 0.3),
            30
        ), 0)
        if rnd.random() < (p_c * 0.5):
            return 'C'
        A = [internal_memory[21][C][0],internal_memory[21][C][1]]
        total = (A[0]) + (A[1]) + 1e-9
        p_share = (A[0]) / total
        if abs((A[0]) - (A[1])) == 0:
            return rnd.choice(['C','D'])
        elif rnd.random() <= p_share:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Hungry':
        if (len(history) + 1) % 25 == 0 and rnd.random() > 0.1:
            return 'D'
        else:
            if not len(history):
                return 'C'
            return history[-1][opp_index]
    elif strategy == 'Uranium':
        if len(history) > 0:
            if history[-1][self_index] == 'D' and rnd.random() > 0.05:
                return 'D'
            if (len(history) + 1) > 2:
                if history[-1][opp_index] == 'D' or history[-1-1][opp_index] == 'D':
                    return 'D'
        return 'C'
    elif strategy == 'Markov':
        if (history[-1][opp_index] == 'C') if len(history) > 0 else True:
            if (history[-1][self_index] == 'C') if len(history) > 0 else True:
                internal_memory[22][0] += 1
            else:
                internal_memory[22][1] += 1
        else:
            if history[-1][self_index] == 'C':
                internal_memory[22][2] += 1
            else:
                internal_memory[22][3] += 1
        p_cc = internal_memory[22][0] / (len(history) + 1)
        p_cd = internal_memory[22][1] / (len(history) + 1)
        p_dc = internal_memory[22][2] / (len(history) + 1)
        p_dd = internal_memory[22][3] / (len(history) + 1)
        if (history[-1][self_index] == 'C') if len(history) > 0 else True:
            total = p_cc + p_dc
            A = p_cc / total
            if rnd.random() <= A:
                return 'C'
            else:
                return 'D'
        else:
            total = p_dd + p_cd
            A = p_dd / total
            if rnd.random() <= A:
                return 'D'
            else:
                return 'C'
    elif strategy == 'Josuah':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'C':
            internal_memory[24][3] += 1
        p_c = internal_memory[24][3] / (len(history))
        if rnd.random() < (internal_memory[24][0] * p_c):
            internal_memory[24][1] = 1
            internal_memory[24][2] = 1
        if internal_memory[24][1] == 1:
            if history[-1][opp_index] == 'D':
                internal_memory[24][0] *= 0.9
                internal_memory[24][1] = 0
                return 'C'
            else:
                if internal_memory[24][2] == 1:
                    internal_memory[24][2] = 0
                    return 'D'
                else:
                    internal_memory[24][2] = 1
                    return 'C'
        else:
            return history[-1][opp_index]
    elif strategy == 'Smart Generous Tit For Tat':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'C':
            internal_memory[25] += 1
        p_c = (internal_memory[25] / (len(history)))
        if rnd.random() < (0.2 * p_c):
            return 'C'
        else:
            return history[-1][opp_index]
    elif strategy == 'Tiny Brain':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'C':
            internal_memory[27][0] += 1
        elif rnd.random() < (0.25 * (internal_memory[27][0] / (len(history) + 1))):
            internal_memory[27][1] += 1
            return 'C'
        p_c = internal_memory[27][0] / (len(history) + 1)
        p_d = 1 - p_c
        p_f = internal_memory[27][1] / (len(history) + 1)
        SH = (history[-1][opp_index] == 'C')
        ST = 1 - SH
        LIN = ((p_c * SH) - (p_d * ST)) + p_f
        A = max(min(LIN,1),0)
        if rnd.random() <= A:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Alan':
        if not len(history):
            return 'C'
        instruction_string = '0:C0|C1,1:C0|D2,2:C1|D3,3:C2|D4,4:D3|D4'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                mode_number, action = item.split(':')
                action_0_str, action_1_str = action.split('|')
                action_0 = [action_0_str[0], action_0_str[1:]]
                action_1 = [action_1_str[0], action_1_str[1:]]
                rule[mode_number] = [action_0, action_1]
            return rule
        instruction = translating(instruction_string)
        alan_rule = instruction[internal_memory[28]][(history[-1][opp_index] == 'D')]
        internal_memory[28] = alan_rule[1]
        return alan_rule[0]
    elif strategy == 'Unpredictable Tit For Tat':
        A = rnd.random()
        if A > 0.9:
            return 'C'
        elif A < 0.1:
            return 'D'
        else:
            if not len(history):
                return 'C'
            return history[-1][opp_index]
    elif strategy == 'Nash':
        cond = ((2 * (history[-2][self_index] == 'D')) + (history[-2][opp_index] == 'D')) if (len(history) + 1) > 2 else 0
        internal_memory[280][cond] *= 0.75
        internal_memory[280][cond] += 0.25 * ((history[-1][opp_index] == 'C') if len(history) > 0 else True)
        reward_cc = self_payoff_matrix['CC']
        reward_dc = self_payoff_matrix['DC']
        punisment_cd = self_payoff_matrix['CD']
        punisment_dd = self_payoff_matrix['DD']
        nash_think_future = 0.55
        c_factor = (reward_cc + punisment_cd) * (1 - nash_think_future)
        d_factor = (reward_dc + punisment_dd) * (1 - nash_think_future)
        c_factor += ((internal_memory[280][0] * reward_cc) + (internal_memory[280][1] * punisment_cd)) * nash_think_future
        d_factor += ((internal_memory[280][2] * reward_dc) + (internal_memory[280][3] * punisment_dd)) * nash_think_future
        E_c = c_factor
        E_d = d_factor
        if E_c > E_d:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Line':
        if (len(history)) < 2:
            if not len(history):
                return 'C'
            return history[-1][opp_index]
        else:
            return history[-1][self_index]
    elif strategy == 'Smart Cat':
        D = 1 - (1 / (len(history) + 1))
        internal_memory[30] *= D
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[30] += 1
            p_d = internal_memory[30] / (len(history) + 1)
            if rnd.random() <= (p_d * 0.6):
                return 'D'
        if not len(history):
            return 'C'
        return history[-1][opp_index]
    elif strategy == 'Cerebrum':
        brain = internal_memory[31][0] if isinstance(internal_memory[31], list) else internal_memory[31]
        brain["day_tick"] = (brain["day_tick"] + 1) % 100
        n = brain["n"]
        adj = brain["adj"]       
        volt = brain["volt"]     
        th = brain["th"]         
        ref = brain["ref"]       
        A_plus = 0.015
        A_minus = 0.018
        tau_plus = 20.0
        tau_minus = 20.0
        current_time = (len(history) + 1)
        last_spike = brain["last_spike_time"]
        brain["CORT"] *= 0.97  
        brain["NOR"]  *= 0.20  
        brain["ADR"]  *= 0.10  
        brain["DA"]   *= 0.50  
        brain["SER"]  *= 0.80  
        brain["END"]  *= 0.70  
        brain["OXY"]  *= 0.85  
        brain["TES"]  *= 0.95  
        brain["VAS"]  *= 0.90  
        brain["GAB"]  *= 0.05  
        brain["ACH"]  *= 0.40  
        brain["GLU"]  *= 0.01  
        brain["MEL"]  *= 0.90  
        opp = history[-1][opp_index] if len(history) > 0 else 'C'
        self_prev = history[-1][self_index] if len(history) > 0 else 'C'
        opp_share = 1.0 if opp == 'C' else 0.0
        opp_steal = 1.0 - opp_share
        self_share = 1.0 if self_prev == 'C' else 0.0
        self_steal = 1.0 - self_share
        exploiting  = (self_prev == 'D' and opp == 'C')
        punished    = (self_prev == 'D' and opp == 'D')
        exploited   = (self_prev == 'C' and opp == 'D')
        cooperating = (self_prev == 'C' and opp == 'C')
        horm_CORT = brain["CORT"]
        horm_NOR = brain["NOR"] 
        horm_ADR = brain["ADR"] 
        horm_DA = brain["DA"]  
        horm_SER = brain["SER"] 
        horm_END = brain["END"] 
        horm_OXY = brain["OXY"] 
        horm_TES = brain["TES"] 
        horm_VAS = brain["VAS"] 
        horm_GAB = brain["GAB"] 
        horm_ACH = brain["ACH"] 
        horm_ADE = brain["ADE"] 
        horm_MEL = brain["MEL"] 
        horm_GLU = brain["GLU"] 
        if brain["is_sleeping"]:
            brain["ADE"] *= 0.70
            brain["MEL"] *= 0.80
            brain["energy"] = min(1000.0, brain["energy"] + 150.0)
            brain["lactate"] *= 0.5
            brain["CORT"] *= 0.80
            brain["NOR"] *= 0.02
            if brain["energy"] > 400.0 and horm_ADE < 0.10 and horm_MEL < 0.10 and brain["day_tick"] <= 70:
                brain["is_sleeping"] = False
            if exploited:
                brain["ADR"] += 0.30
                brain["NOR"] += 0.30
                brain["GLU"] += 0.30
                brain["MEL"] = max(0.0, brain["MEL"] - 0.40)
                brain["ADE"] = max(0.0, brain["ADE"] - 0.20)
            return 'C'
        cerebrum_eyes = [('C', 'C') for _ in range(10)] + history
        inp_list = [val for i in range(10) for val in [(cerebrum_eyes[-1 - (9 - i)][opp_index] == 'C') * (brain["decay factor"][i]), (cerebrum_eyes[-1 - (9 - i)][opp_index] == 'D') * (brain["decay factor"][i]), (cerebrum_eyes[-1 - (9 - i)][self_index] == 'C') * (brain["decay factor"][i]), (cerebrum_eyes[-1 - (9 - i)][self_index] == 'D') * (brain["decay factor"][i])]] 
        inp = np.array(inp_list, dtype=np.float64)
        tot_spk = 0
        gf = (0.35 + (horm_ACH * 0.1)) * ((1.0 - (horm_SER * 0.05)) * (1.0 - (horm_MEL * 0.4)) + (1 + (horm_NOR * 0.1)))
        for _ in range(4):
            old_volt = volt.copy()
            volt[:40] += inp * gf
            ref_active_mask = ref > 0
            ref[ref_active_mask] -= 1
            total_incoming = np.dot(old_volt, adj) * horm_GLU
            noise = np.random.uniform(-0.08 * horm_CORT, 0.08 * horm_CORT, size=n)
            non_ref_mask = ~ref_active_mask
            volt[non_ref_mask] += total_incoming[non_ref_mask] + noise[non_ref_mask]
            volt[non_ref_mask] *= 0.86
            volt[non_ref_mask] = np.clip(volt[non_ref_mask], -3.0, 3.0)
            firing_mask = (ref == 0) & (volt >= th)
            if np.any(firing_mask):
                volt[firing_mask] = 0.0
                ref[firing_mask] = 2
                tot_spk += np.sum(firing_mask)
                firing_indices = np.where(firing_mask)[0]
                dt = current_time - last_spike
                pre_trace = np.zeros(n)
                valid_pre = last_spike != -np.inf
                pre_trace[valid_pre] = np.exp(
                    -dt[valid_pre] / tau_plus
                )
                post_trace = np.zeros(n)
                valid_post = last_spike != -np.inf
                post_trace[valid_post] = np.exp(
                    -dt[valid_post] / tau_minus
                )
                if len(firing_indices) > 0:
                    adj[:, firing_indices] += (
                        A_plus
                        * pre_trace[:, None]
                        * horm_DA
                    )
                    adj[firing_indices, :] -= (
                        A_minus
                        * post_trace[None, :]
                        * horm_DA
                    )
                    adj = np.clip(adj, -1.0, 1.0)
                last_spike[firing_indices] = current_time    
            th_delta = np.where(volt > th, 0.02, -0.08)
            th = np.clip(th + th_delta, 0.6, 2.5)
        brain["energy"] -= (tot_spk * 0.04)
        brain["lactate"] = min(2.0, brain["lactate"] + (tot_spk * 0.0015))
        brain["GLU"] = min(1.0, brain["GLU"] + (tot_spk * 0.01))
        brain["MEL"] = min(1.0, brain["MEL"] + 0.15) if brain["day_tick"] > 70 else max(0.0, brain["MEL"] - 0.1)
        brain["ADE"] = min(1.0, brain["ADE"] + 0.01)
        if brain["energy"] <= 0 or horm_ADE > 0.95:
            brain["is_sleeping"] = True
        brain["volt"] = volt
        brain["th"] = th
        brain["ref"] = ref
        half_n = n // 2
        share_signal = np.sum(np.maximum(volt[:half_n], 0))
        steal_signal = np.sum(np.maximum(volt[half_n:], 0))
        if exploiting:
            brain["DA"] += 0.05
            brain["CORT"] -= 0.03
            brain["energy"] += 50.0
        elif exploited:
            brain["DA"] -= 0.06
            brain["CORT"] += 0.08
            brain["energy"] -= 40.0
        elif punished:
            if horm_CORT > 0.6:
                brain["DA"] -= 0.01
                brain["CORT"] += 0.05
            else:
                brain["DA"] += 0.015
                brain["CORT"] += 0.02
            brain["energy"] -= 10.0
        elif cooperating:
            brain["DA"] += 0.02
            brain["CORT"] -= 0.03
            brain["energy"] += 30.0
        if exploiting:
            brain["TES"] += 0.05  
            brain["OXY"] -= 0.04  
            brain["SER"] -= 0.02  
        elif exploited:
            brain["ADR"] += 0.06  
            brain["OXY"] -= 0.06  
        elif cooperating:
            brain["OXY"] += 0.07  
            brain["SER"] += 0.05  
        elif punished:
            brain["NOR"] += 0.06  
            brain["ADR"] += 0.04  
            brain["SER"] -= 0.04  
        lead = score[self_index] - score[opp_index]
        delta_lead = (lead - brain["lead"])
        if brain["lead"] != lead:
            brain["GLU"] += 0.10 * abs(delta_lead)
        if delta_lead > 0:
            brain["DA"] += 0.12 * delta_lead
        else:
            brain["CORT"] += 0.10 * abs(delta_lead)
            brain["ADR"] += 0.12 * abs(delta_lead)
        brain["lead"] = lead
        energy_ratio = max(0.0, min(1.0, ((brain["energy"] - 450.0) / 100.0)))
        energy_deficit = 1.0 - energy_ratio
        metabolic_stress = energy_deficit * energy_deficit
        brain["CORT"] += metabolic_stress * 0.08
        brain["ADR"] += metabolic_stress * 0.10
        brain["GLU"] += metabolic_stress * 0.05
        brain["DA"] -= metabolic_stress * 0.05
        brain["GLU"] += ((horm_NOR * 0.04) + (horm_ADR * 0.03) + (horm_ACH * 0.02)) - ((horm_GAB * 0.05) + (horm_MEL * 0.04))
        brain["GAB"] += ((horm_SER * 0.04) + (horm_END * 0.03) + (horm_MEL * 0.05)) - ((horm_GLU * 0.04) + (horm_NOR * 0.03) + (horm_ADR * 0.02))
        brain["CORT"] += ((horm_ADR * 0.04) + (horm_NOR * 0.03) + (horm_GLU * 0.02)) - ((horm_OXY * 0.04) + (horm_SER * 0.03) + (horm_MEL * 0.03))
        brain["ADR"] += ((horm_NOR * 0.04) + (horm_CORT * 0.02) + (horm_GLU * 0.02)) - ((horm_GAB * 0.05) + (horm_MEL * 0.04))
        brain["NOR"] += ((horm_ADR * 0.03) + (horm_GLU * 0.02)) - ((horm_OXY * 0.03) + (horm_GAB * 0.04) + (horm_MEL * 0.05))
        brain["DA"] += ((horm_END * 0.03) + (horm_ACH * 0.03) + (horm_GLU * 0.02)) - ((horm_CORT * 0.04) + (horm_GAB * 0.02) + (horm_ADE * 0.03))
        brain["SER"] += ((horm_GAB * 0.03) + (horm_OXY * 0.03) + (horm_END * 0.02)) - ((horm_CORT * 0.04) + (horm_ADR * 0.02))
        brain["ACH"] += ((horm_DA * 0.03) + (horm_GLU * 0.03) + (horm_NOR * 0.02)) - ((horm_GAB * 0.03) + (horm_MEL * 0.04))
        brain["END"] += ((horm_ADR * 0.03) + (horm_DA * 0.02)) - ((horm_CORT * 0.03) + (horm_GAB * 0.01))
        brain["OXY"] += ((horm_SER * 0.04) + (horm_DA * 0.02)) - ((horm_CORT * 0.05) + (horm_TES * 0.03) + (horm_VAS * 0.02))
        brain["TES"] += ((horm_DA * 0.03) + (horm_ADR * 0.02) + (horm_VAS * 0.02)) - ((horm_OXY * 0.04) + (horm_SER * 0.02))
        brain["VAS"] += ((horm_CORT * 0.04) + (horm_TES * 0.03) + (horm_NOR * 0.02)) - ((horm_OXY * 0.03) + (horm_SER * 0.02))
        brain["ADE"] += ((horm_GLU * 0.04) + (horm_ADR * 0.02) + (horm_NOR * 0.02)) - ((horm_MEL * 0.15))
        brain["MEL"] += ((horm_ADE * 0.05) + (horm_SER * 0.02)) - ((horm_GLU * 0.04) + (horm_NOR * 0.04))
        if cooperating:
            brain["ACH"] = min(1.0, brain["ACH"] + 0.05)
        elif exploited:
            brain["ACH"] = max(0.1, brain["ACH"] - 0.12)  
            brain["GLU"] = min(1.0, brain["GLU"] + 0.20)
        if cooperating and brain["energy"] > 1000.0:
            brain["GAB"] = min(1.0, brain["GAB"] + 0.03)
            brain["energy"] = max(1000.0, brain["energy"] - 15.0)
        for key in ["DA", "CORT", "SER", "OXY", "ADR", "NOR", "TES", "GAB", "END", "ACH", "VAS", "GLU", "ADE", "MEL"]:
            brain[key] = float(np.clip(brain[key], 0.05 if key in ["DA", "CORT"] else 0.0, 1.0))
        stress = horm_CORT
        reward = horm_DA
        calm = horm_SER + horm_OXY
        if stress > 0.75:
            personality_mode = "panic"
        elif reward > 0.75:
            personality_mode = "hungry"
        elif calm > 0.7:
            personality_mode = "stable"
        else:
            personality_mode = "neutral"
        if horm_DA > 0.82 and n < brain["max_n"]:
            if rnd.random() < 0.40:
                new_n = n + 1
                new_adj = np.pad(adj, ((0, 1), (0, 1)), mode='constant')
                new_adj[:-1, -1] = np.random.uniform(-0.15, 0.15, n)
                new_adj[-1, :] = np.random.uniform(-0.15, 0.15, new_n)
                brain["adj"] = new_adj
                brain["volt"] = np.append(volt, 0.0)
                brain["th"] = np.append(th, rnd.uniform(0.9, 1.1))
                brain["ref"] = np.append(ref, 0)
                brain["n"] = new_n
                brain["last_spike_time"] = np.append(
                    brain["last_spike_time"],
                    -np.inf
                )
        elif (horm_CORT > 0.78 or horm_DA < 0.15) and n > brain["min_n"]:
            if rnd.random() < 0.25:
                conn_strength = np.sum(np.abs(adj[40:, :]), axis=1) + np.sum(np.abs(adj[:, 40:]), axis=0)
                activity = np.abs(volt[40:])
                weakness_scores = conn_strength + activity
                weakest_idx = np.argmin(weakness_scores) + 4
                brain["last_spike_time"] = np.delete(
                brain["last_spike_time"],
                weakest_idx
                )
                brain["adj"] = np.delete(np.delete(adj, weakest_idx, axis=0), weakest_idx, axis=1)
                brain["volt"] = np.delete(volt, weakest_idx)
                brain["th"] = np.delete(th, weakest_idx)
                brain["ref"] = np.delete(ref, weakest_idx)
                brain["n"] -= 1
        mutual = (self_prev == opp)
        if exploited:
            brain["a"] += 0.01
        elif exploiting:
            brain["a"] -= 0.008
        elif mutual:
            brain["a"] -= 0.002
        brain["a"] = float(np.clip(brain["a"], 0.0, 1.0))
        a = brain["a"]
        social_bias = 1.5 * (horm_SER + horm_OXY + horm_END + horm_GAB + horm_VAS + horm_ACH)
        aggressive_bias = 1.5 * (horm_ADR + horm_NOR + horm_TES + horm_CORT + horm_DA)
        social_bias *= (1.0 - horm_ACH * 0.3)
        aggressive_bias *= (1.0 + horm_ACH * 0.2)
        p_share = share_signal * (((1.0 - a) + (a * opp_share)) * social_bias)
        p_steal = steal_signal * ((a * opp_steal) * aggressive_bias)
        if personality_mode == "panic":
            p_steal *= 1.4; p_share *= 0.8
        elif personality_mode == "hungry":
            p_steal *= 1.3; p_share *= 1.1
        elif personality_mode == "stable":
            p_share *= 1.3; p_steal *= 0.9
        p_share = float(np.clip(p_share, 0.0, 1.0))
        p_steal = float(np.clip(p_steal, 0.0, 1.0))
        brain["a"] = float(np.clip(brain["a"] + 0.005 * (horm_CORT - horm_DA), 0.0, 1.0))
        if p_share >= p_steal:
            return 'C'
        else:
            return 'D'
    elif strategy == 'SURPRISE ATTACK!':
        attack_p = 1 - max(min(((tournament_avg_last_round / 2) / (len(history) + 1)),1),0)
        if (rnd.random() < attack_p) or (internal_memory[32] == 1):
            if internal_memory[32] == 0:
                internal_memory[32] = 1
            return 'D'
        else:
            if not len(history):
                return 'C'
            return history[-1][opp_index]
    elif strategy == 'Sumobox Fighter':
        brain = internal_memory[34][0]
        another_variable = internal_memory[34][1]
        another_variable[0] -= 1
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                another_variable[2] += 1
        A = 5
        sumo_eyes = [('C', 'C') for _ in range(A)] + history
        if len(brain) != 0:
            if sumo_eyes[-1][opp_index] == 'C':
                if sumo_eyes[-1][self_index] == 'C':
                    brain[another_variable[1]][1] += self_payoff_matrix['CC']
                else:
                    brain[another_variable[1]][1] += self_payoff_matrix['DC']
            else:
                if sumo_eyes[-1][self_index] == 'C':
                    brain[another_variable[1]][1] -= 4 * (self_payoff_matrix['DD'] - self_payoff_matrix['CD'])
                else:
                    brain[another_variable[1]][1] += self_payoff_matrix['DD']
            if brain[another_variable[1]][1] <= 0 and another_variable[0] <= 0:
                new_output = "".join([rnd.choice(["C", "D"]) for _ in range(A)])
                brain[another_variable[1]] = [new_output,100]
            if brain[another_variable[1]][1] >= 200 and another_variable[0] <= 0:
                if rnd.random() <= (another_variable[2] / (len(history) + 1)):
                    new_choice = "D"
                else:
                    new_choice = "C"
                idx = rnd.randint(0,A-1)
                new_output = brain[another_variable[1]][0][:idx] + new_choice + brain[another_variable[1]][0][idx+1:]
                brain[another_variable[1]] = [new_output,100]
        if another_variable[0] <= 0:
            inp = tuple(["".join([sumo_eyes[-i][opp_index] for i in range(A, 0, -1)]),"".join([sumo_eyes[-j][self_index] for j in range(A, 0, -1)])]) 
            another_variable[0] = A
            another_variable[1] = inp
            def similarity(new_key,old_key):
                new_key_vector_opp = [1 if new_key[0][S1] == "C" else -1 for S1 in range(A)]
                old_key_vector_opp = [1 if old_key[0][S2] == "C" else -1 for S2 in range(A)]
                new_key_vector_self = [1 if new_key[1][S1] == "C" else -1 for S1 in range(A)]
                old_key_vector_self = [1 if old_key[1][S2] == "C" else -1 for S2 in range(A)]
                similarity_val = 0
                for SRT in range(A):
                    similarity_val += ((new_key_vector_opp[SRT] * old_key_vector_opp[SRT]) + (new_key_vector_self[SRT] * old_key_vector_self[SRT]))
                return ((similarity_val + A) / 2) / (A * 2)
            if brain.get(inp) == None:
                if len(brain) != 0:
                    target_space = min(brain, key=lambda k: brain[k][1])
                    is_same = similarity(inp,target_space)
                    new_output = "".join([rnd.choice(["C", "D"]) for _ in range(A)])
                    if is_same >= rnd.random():
                        brain[target_space] = [new_output,100]
                        another_variable[1] = target_space
                    else:
                        brain[inp] = [new_output,100]
                        another_variable[1] = inp
                else:
                    new_output = "".join([rnd.choice(["C", "D"]) for _ in range(A)])
                    brain[inp] = [new_output,100]
                    another_variable[1] = inp
        idx = A - another_variable[0]
        idx = max(0, min(idx, (A-1)))
        if brain[another_variable[1]][0][idx] == "C":
            return 'C'
        else:
            return 'D'
    elif strategy == 'Smart Snake':
        brain = internal_memory[36]
        smart_snake_eyes = [('C', 'C') for _ in range(5)] + history
        if len(history) > 0:
            dis = [1,0.8,0.6,0.4,0.2]
            if smart_snake_eyes[-1][opp_index] == 'C':
                if smart_snake_eyes[-1][self_index] == 'C':
                    for learn in range(5):
                        brain[0][learn] = max(min((brain[0][learn] + (0.333 * (self_payoff_matrix['CC'] / 3) * dis[learn])),1),-1)
                else:
                    for learn in range(5):
                        brain[1][learn] = max(min((brain[1][learn] + (0.5 * (self_payoff_matrix['DC'] / 5) * dis[learn])),1),-1)
            else:
                if smart_snake_eyes[-1][self_index] == 'C':
                    for learn in range(5):
                        brain[0][learn] = max(min((brain[0][learn] - (0.4 * (self_payoff_matrix['DD'] - self_payoff_matrix['CD']) * dis[learn])),1),-1)
                else:
                    for learn in range(5):
                        brain[1][learn] = max(min((brain[1][learn] + (0.1 * (self_payoff_matrix['DD']) * dis[learn])),1),-1)
        C_input = [(smart_snake_eyes[-i][opp_index] == 'C') for i in range(5, 0, -1)] 
        D_input = [(smart_snake_eyes[-i][opp_index] == 'D') for i in range(5, 0, -1)] 
        c_factor = (
            (C_input[0] * brain[0][0]) +
            ((C_input[1] * brain[0][1]) * 0.8) +
            ((C_input[2] * brain[0][2]) * 0.6) +
            ((C_input[3] * brain[0][3]) * 0.4) +
            ((C_input[4] * brain[0][4]) * 0.2)
        )
        d_factor = (
            (D_input[0] * brain[1][0]) +
            ((D_input[1] * brain[1][1]) * 0.8) +
            ((D_input[2] * brain[1][2]) * 0.6) +
            ((D_input[3] * brain[1][3]) * 0.4) +
            ((D_input[4] * brain[1][4]) * 0.2)
        )
        if c_factor == d_factor:
            return rnd.choice(['C','D'])
        elif c_factor > d_factor:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Theseus':
        another_factor = internal_memory[38][1]
        brain = internal_memory[38][0]
        theseus_eyes = [('C', 'C') for _ in range(5)] + history
        another_factor[1] *= 0.9
        if theseus_eyes[-1][opp_index] == 'D':
            another_factor[1] += 0.1
        if len(history) > 0:
            if theseus_eyes[-1][opp_index] == 'C':
                if theseus_eyes[-1][self_index] == 'C':
                    brain[another_factor[0]] = max(min(brain[another_factor[0]] + (0.333 * (self_payoff_matrix['CC'] / 3) * (1 - another_factor[1])),1),0)
                else:
                    brain[another_factor[0]] = max(min(brain[another_factor[0]] - (0.5 * (self_payoff_matrix['DC'] / 5) * another_factor[1]),1),0)
            else:
                if theseus_eyes[-1][self_index] == 'C':
                    brain[another_factor[0]] = max(min(brain[another_factor[0]] - (0.4 * (self_payoff_matrix['DD'] - self_payoff_matrix['CD']) * another_factor[1]),1),0)
                else:
                    brain[another_factor[0]] = max(min(brain[another_factor[0]] - (0.1 * self_payoff_matrix['DD'] * another_factor[1]),1),0)
        inp = tuple(["".join([theseus_eyes[-i][opp_index] for i in range(5, 0, -1)]),"".join([theseus_eyes[-j][self_index] for j in range(5, 0, -1)])]) 
        SH = (theseus_eyes[-1][opp_index] == 'C')
        if brain.get(inp) == None:
            brain[inp] = ((1 - another_factor[1]) * 0.9) + (SH * 0.1)
        another_factor[0] = inp
        if rnd.random() <= brain[inp]:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Social Average':
        tm = internal_memory[39]
        if not len(history):
            return 'C'
        opp = history[-1][opp_index]
        self_prev = history[-1][self_index]
        if opp == 'C':
            tm[0] += 1
            tm[2] += 0.08
            tm[3] -= 0.05
        else:
            tm[1] += 1
            tm[2] -= 0.12
            tm[3] += 0.10
        tm[2] = max(0, min(1, tm[2]))
        tm[3] = max(0, min(1, tm[3]))
        total = tm[0] + tm[1] + 1e-9
        p_coop = tm[0] / total
        p_def = tm[1] / total
        if p_coop > 0.7:
            if rnd.random() < 0.9:
                return 'C'
            else:
                return 'D'
        elif p_def > 0.65:
            if opp == 'D':
                return 'D'
            else:
                if rnd.random() < 0.2:
                    return 'D'
                return 'C'
        decision = opp
        if opp == self_prev:
            if rnd.random() < 0.75:
                decision = 'C'
        if tm[3] > 0.6:
            if rnd.random() < tm[3]:
                decision = 'D'
        if tm[2] > 0.75:
            if rnd.random() < tm[2]:
                decision = 'C'
        if rnd.random() < 0.08:
            decision = rnd.choice(['C','D'])
        if (
            opp == 'C'
            and history[-1-1][opp_index] == 'C'
            and rnd.random() < 0.15
        ):
            tm[4] += 1
            return 'D'
        if tm[5] > 0:
            tm[5] -= 1
            return 'C'
        if (
            opp == 'D'
            and self_prev == 'C'
            and rnd.random() < 0.12
        ):
            tm[5] = 2
            return 'C'
        return decision
    elif strategy == 'Capitalist':
        tm = internal_memory[40]
        if not len(history):
            return 'C'
        opp = history[-1][opp_index]
        if opp == 'C':
            tm[1] += 0.05
        else:
            tm[1] -= 0.12
        tm[1] = max(0,min(1,tm[1]))
        tm[0] += 0.002
        tm[0] = min(tm[0],1)
        exploit_chance = (
            (tm[0] * 0.6) +
            ((1 - tm[1]) * 0.4)
        ) * 0.125
        if opp == 'C':
            if rnd.random() < exploit_chance:
                tm[2] += 1
                return 'D'
            else:
                return 'C'
        else:
            if rnd.random() < (tm[1] * 0.5):
                return 'C'
            return 'D'
    elif strategy == 'Communalist':
        tm = internal_memory[41]
        if not len(history):
            return 'C'
        opp = history[-1][opp_index]
        if opp == 'C':
            tm[0] += 1
            tm[2] += 0.04
        else:
            tm[1] += 1
            tm[2] -= 0.08
        tm[2] = max(0,min(1,tm[2]))
        total = tm[0] + tm[1] + 1e-9
        social_stability = tm[0] / total
        if social_stability < 0.45:
            if opp == 'D':
                return 'D'
        if rnd.random() < tm[2]:
            return 'C'
        if opp == 'D':
            return 'D'
        return 'C'
    elif strategy == 'Tit For Tat But Not Too Forgiving':
        opp_prev = history[-1-1][opp_index] if (len(history) + 1) > 2 else 'C'
        opp = history[-1][opp_index] if len(history) > 0 else 'C'
        self = history[-1][self_index] if len(history) > 0 else 'C'
        if opp == 'C':
            if self == 'C':
                return 'C'
            else:
                return rnd.choice(['C','D'])
        else:
            if self == 'C':
                return 'D'
            else:
                if opp_prev == 'D':
                    return 'D'
                else:
                    return rnd.choice(['C','D'])
    elif strategy == 'Spectrum Attacker':
        A = max(min((((len(history) + 1) - (tournament_avg_last_round / 2)) / (tournament_avg_last_round / 2)),1),0)
        if rnd.random() <= A:
            return 'D'
        else:
            if not len(history):
                return 'C'
            return history[-1][opp_index]
    elif strategy == 'Smooth Grudger':
        opp = history[-1][opp_index] if len(history) > 0 else 'C'
        internal_memory[42] *= 0.9
        if opp == 'D':
            internal_memory[42] += 0.1
        if rnd.random() < internal_memory[42]:
            return 'D'
        else:
            return opp
    elif strategy == 'Forest-RND':
        brain = internal_memory[43][0]
        trees_input = [internal_memory[43][1],internal_memory[43][2]]
        old_output = internal_memory[43][3]
        old_inp = internal_memory[43][4]
        rating = internal_memory[43][5]
        forest_eyes = [('C', 'C') for _ in range(9)] + history
        share_signal = 0
        steal_signal = 0
        if len(history) > 0:
            R_or_P = None
            if forest_eyes[-1][opp_index] == 'C':
                if forest_eyes[-1][self_index] == 'C':
                    R_or_P = [1,3,1,-1]
                else:
                    R_or_P = [4,1,-1,1]
            else:
                if forest_eyes[-1][self_index] == 'C':
                    R_or_P = [8,1,-2,1]
                else:
                    R_or_P = [4,1,-1,0]
            for n in range(len(brain)):
                if old_output[n]:
                    trees_input[0][n] = (trees_input[0][n] + rnd.randint(-R_or_P[0], R_or_P[0])) % 9
                    trees_input[1][n] = (trees_input[1][n] + rnd.randint(-R_or_P[0], R_or_P[0])) % 9
                    rating[n] += R_or_P[2]
                else:
                    trees_input[0][n] = (trees_input[0][n] + rnd.randint(-R_or_P[1], R_or_P[1])) % 9
                    trees_input[1][n] = (trees_input[1][n] + rnd.randint(-R_or_P[1], R_or_P[1])) % 9
                    rating[n] += R_or_P[3]
                if rating[n] <= -5:
                    if brain[n][old_inp[n]] == "C":
                        brain[n][old_inp[n]] = "D"
                    else:
                        brain[n][old_inp[n]] = "C"
                    rating[n] = 0
                elif rating[n] >= 5:
                    trees_input[0][n] = (trees_input[0][n] + rnd.randint(-1, 1)) % 9
                    trees_input[1][n] = (trees_input[1][n] + rnd.randint(-1, 1)) % 9
                    rating[n] = 0
                rating[n] = max(min(rating[n],5),-5)
        for n in range(len(brain)):
            inp1 = forest_eyes[-1 - trees_input[0][n]][opp_index]
            inp2 = forest_eyes[-1 - trees_input[1][n]][self_index]
            inp = (inp1, inp2)
            if brain[n].get(inp) == None:
                brain[n][inp] = rnd.choice(["C","D"])
            if brain[n][inp] == "C":
                share_signal += 1
                old_output[n] = True
            else:
                steal_signal += 1
                old_output[n] = False
            old_inp[n] = inp
        if share_signal == steal_signal:
            return forest_eyes[-1][opp_index]
        elif share_signal > steal_signal:
            return 'C'
        else:
            return 'D'
    elif strategy == '3-Tree Of Prediction':
        primary_tree = internal_memory[44][0]
        secondary_tree = internal_memory[44][1]
        ttop_eyes = [('C', 'C')] + history
        if len(history) > 0:
            R_or_P = None
            if ttop_eyes[-1][opp_index] == 'C':
                R_or_P = 'reward'
            else:
                R_or_P = 'punisment'
            self = ttop_eyes[-1][self_index]
            rev_self = "C" if ttop_eyes[-1][self_index] == 'D' else "D"
            if R_or_P == 'reward':
                if primary_tree[internal_memory[44][2][0]] != self:
                    primary_tree[internal_memory[44][2][0]] = rev_self
            else:
                if primary_tree[internal_memory[44][2][0]] == self:
                    primary_tree[internal_memory[44][2][0]] = rev_self
            for n in range(2):
                if R_or_P == 'reward':
                    if secondary_tree[n][internal_memory[44][2][n+1]] != self:
                        secondary_tree[n][internal_memory[44][2][n+1]] = rev_self
                else:
                    if secondary_tree[n][internal_memory[44][2][n+1]] == self:
                        secondary_tree[n][internal_memory[44][2][n+1]] = rev_self
        F_inp1 = ttop_eyes[-1][opp_index]
        F_inp2 = ttop_eyes[-1][self_index]
        F_inp = (F_inp1, F_inp2)
        if primary_tree.get(F_inp) == None:
            primary_tree[F_inp] = rnd.choice(["C","D"])
        first_choice = primary_tree[F_inp]
        internal_memory[44][2][0] = F_inp
        share_signal = 0
        steal_signal = 0
        S_inp = [("C", first_choice), ("D", first_choice)]
        for n in range(2):
            if secondary_tree[n].get(S_inp[n]) == None:
                secondary_tree[n][S_inp[n]] = rnd.choice(["C","D"])
            if secondary_tree[n][S_inp[n]] == "C":
                share_signal += 1
            else:
                steal_signal += 1
            internal_memory[44][2][n+1] = S_inp[n]
        if share_signal == steal_signal:
            return ttop_eyes[-1][opp_index]
        elif share_signal > steal_signal:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Bayes Attariq Version':
        tm = internal_memory[45]
        self = history[-1][self_index] if len(history) > 0 else 'C'
        opp = history[-1][opp_index] if len(history) > 0 else 'C'
        if self == 'C':
            tm[0] += 1
            if opp == 'C':
                tm[1] += 1
        else:
            tm[2] += 1
            if opp == 'C':
                tm[3] += 1
        SH = (opp == 'C')
        a = 0.1
        P1 = ((tm[1] / (tm[0] + 1e-9)) * (1 - a)) + (SH * a)
        P2 = ((tm[3] / (tm[2] + 1e-9)) * (1 - a)) + (SH * a)
        reward_cc = self_payoff_matrix['CC']
        reward_dc = self_payoff_matrix['DC']
        punisment_cd = self_payoff_matrix['CD']
        punisment_dd = self_payoff_matrix['DD']
        c_factor = (P1 * reward_cc) + ((1-P1) * punisment_cd)
        d_factor = (P2 * reward_dc) + ((1-P1) * punisment_dd)
        if c_factor >= d_factor:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Naive Q-Learner':
        brain = internal_memory[46][0]
        another_factor = internal_memory[46][1]
        A = min(12, len(history)) - 1
        opp_inp = tuple(['C' for _ in range(12 - min(12, len(history)))] + [history[-i][opp_index] for i in range(min(12, len(history)), 0, -1)]) 
        self_inp = tuple(['C' for _ in range(12 - min(12, len(history)))] + [history[-i][self_index] for i in range(min(12, len(history)), 0, -1)]) 
        inp = (opp_inp, self_inp)
        if brain.get(inp) == None:
            brain[inp] = [0,0]
        if len(history) > 0 and another_factor[0][1] != 0.5:
            reward = 0
            if history[-1][opp_index] == 'C':
                if history[-1][self_index] == 'C':
                    reward = self_payoff_matrix['CC']
                else:
                    reward = self_payoff_matrix['DC']
            else:
                if history[-1][self_index] == 'C':
                    reward = self_payoff_matrix['CD']
                else:
                    reward = self_payoff_matrix['DD']
            c_factor = brain[inp][0]
            d_factor = brain[inp][1]
            a = 0.1
            gamma = 0.5
            Q_old = brain[another_factor[0][0]][another_factor[0][1]]
            Q_new = Q_old + (a * (reward + ((gamma * max(c_factor, d_factor)) - Q_old)))
            brain[another_factor[0][0]][another_factor[0][1]] = Q_new
        if rnd.random() <= another_factor[1]:
            Action = rnd_prob(0.5)
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        decay = 0.75
        min_epsilon = 0
        another_factor[1] = max(another_factor[1] * decay, min_epsilon)
        if brain[inp][0] == brain[inp][1]:
            Action = rnd_prob(0.5)
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        best_choice = max(brain[inp])
        if best_choice == brain[inp][0]:
            another_factor[0] = [inp, 0]
            return 'C'
        else:
            another_factor[0] = [inp, 1]
            return 'D'
    elif strategy == 'Stupidiot':
        if (len(history)) < ((50 / 200) * tournament_avg_last_round):
            if len(history) > 0:
                if history[-1][opp_index] == 'D':
                    return 'D'
                if history[-1][self_index] == 'D':
                    if (len(history) + 1) > 2:
                        if history[-1-1][self_index] == 'D':
                            return 'C'
            return 'D'
            return rnd.choice(['C','D'])
        elif (len(history)) < ((100 / 200) * tournament_avg_last_round):
            return 'D'
        else:
            if rnd.random() <= 0.1:
                return 'D'
            if history[-1][opp_index] == 'D':
                internal_memory[47] = 1
            if internal_memory[47] == 1:
                return 'D'
            else:
                return 'C'
    elif strategy == 'Smart-Clever':
        if (history[-1][opp_index] == 'C') if len(history) > 0 else True:
            win_or_lose = 'win'
        else:
            win_or_lose = 'lose'
        if win_or_lose == 'lose':
            if history[-1][self_index] == 'C':
                internal_memory[48] = 1 - internal_memory[48]
        if internal_memory[48] == 1:
            if not len(history):
                return 'C'
            return history[-1][opp_index]
        else:
            if not len(history):
                return 'C'
            return history[-1][self_index] if self_payoff_matrix[history[-1][self_index] + history[-1][opp_index]] >= self_avg_payoff_matrix else flip(history[-1][self_index])
    elif strategy == 'Half Sin Square':
        def half_sin_square(x):
            pi = mth.pi
            half_sin_val = mth.sin((x * pi) / 2)
            return half_sin_val * half_sin_val
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'C':
            internal_memory[50] += 1
        p_c = internal_memory[50] / (len(history))
        c_factor = half_sin_square(p_c)
        if rnd.random() <= c_factor:
            return 'C'
        else:
            return 'D'
    elif strategy == 'TF2T & 2TFT':
        internal_memory[51][0] = rnd.randint(0,1)
        if internal_memory[51][0] == 1:
            internal_memory[51][1] *= 0.5
            if ((history[-1][opp_index] == 'D') if len(history) > 0 else False):
                internal_memory[51][1] += 0.5
                return 'D'
            else:
                if rnd.random() <= (0.5 * internal_memory[51][1]):
                    if ((history[-1-1][opp_index] == 'D') if (len(history) + 1) > 2 else False):
                        return 'D'
                    else:
                        return 'C'
                else:
                    return 'C'
        else:
            internal_memory[51][1] *= 0.5
            if ((history[-1][opp_index] == 'D') if len(history) > 0 else False):
                internal_memory[51][1] += 0.5
                if rnd.random() <= (0.5 * (1 - internal_memory[51][1])):
                    if ((history[-1-1][opp_index] == 'D') if (len(history) + 1) > 2 else False):
                        return 'D'
                    else:
                        return 'C'
                else:
                    return 'D'
            else:
                return 'C'
    elif strategy == 'Psycho Attariq Version':
        if not len(history):
            return 'C'
        if internal_memory[52] == 0:
            if history[-1][opp_index] == 'D':
                internal_memory[52] = 1
        if internal_memory[52] == 1:
            if rnd.random() < (0.25 * max(min((((len(history) + 1) - (tournament_avg_last_round / 2)) / (tournament_avg_last_round / 2)), 1), 0)):
                return 'D'
            else:
                return history[-1][opp_index]
        else:
            return history[-1][opp_index]
    elif strategy == 'Socio':
        if not len(history):
            return 'C'
        if internal_memory[53][0] == 0:
            if history[-1][opp_index] == 'D':
                internal_memory[53][0] = 1
                return 'D'
        if internal_memory[53][0] == 1:
            if history[-1][opp_index] == 'D':
                internal_memory[53][1] = 1
            if internal_memory[53][1] == 1:
                return 'D'
            else:
                return rnd.choice(['C','D'])
        else:
            return 'C'
    elif strategy == 'Isolated Tit For Tat':
        if rnd.random() <= 0.1:
            internal_memory[54][1] = 1 - internal_memory[54][1]
            internal_memory[54][0] = (history[-1][opp_index] == 'C') if len(history) > 0 else True
        if internal_memory[54][1] == 1:
            if internal_memory[54][0] == 1:
                return 'C'
            else:
                return 'D'
        else:
            if not len(history):
                return 'C'
            return history[-1][opp_index]
    elif strategy == 'Isolated Clan':
        if rnd.random() <= 0.1:
            internal_memory[55][1] = 1 - internal_memory[55][1]
            internal_memory[55][0] = max(min(((history[-1][opp_index] == 'C') if len(history) > 0 else True) + (rnd.uniform(-0.1,0.1)),1),0)
        if internal_memory[55][1] == 1:
            internal_memory[55][0] = max(min(internal_memory[55][0] + rnd.uniform(-0.1,0.1),1),0)
            if rnd.random() <= internal_memory[55][0]:
                return 'C'
            else:
                return 'D'
        else:
            if not len(history):
                return 'C'
            return history[-1][opp_index]
    elif strategy == 'Neil A':
        if ((history[-1][opp_index] == 'C') if len(history) > 0 else True):
            internal_memory[56] *= 0.9
        else:
            internal_memory[56] = max(min(internal_memory[56] + 1,2),0)
        A = rnd.random()
        if A > internal_memory[56]:
            return 'C'
        elif A > max(internal_memory[56] - 1, 0):
            return 'D'
        else:
            if history[-1][opp_index] == 'D':
                return 'D'
            else:
                return rnd.choice(['C','D'])
    elif strategy == 'Cult Leader':
        leader_key = [0,0,0,1,1]
        others_key = {
                    (0,0,0,1,1):'leader',
                    (0,0,1,0,1):'bishop',
                    (0,1,0,0,1):'citizen',
                    }
        if ((len(history)) < 6) and (len(history) > 0):
            if history[-1][opp_index] == 'C':
                internal_memory[57][(len(history) + 1) - 2] = 0
            else:
                internal_memory[57][(len(history) + 1) - 2] = 1
        if (len(history) + 1) < 6:
            if leader_key[(len(history))] == 0:
                return 'C'
            else:
                return 'D'
        else:
            if (len(history)) < 7:
                return 'C'
            others_pattern = others_key.get(tuple(internal_memory[57]))
            if others_pattern == None:
                if rnd.random() < 0.2:
                    return 'C'
                else:
                    return history[-1][opp_index]
            else:
                if others_pattern == 'leader':
                    return 'C'
                else:
                    return 'D'
    elif strategy == 'Cult Bishop A':
        bishop_key = [0,0,1,0,1]
        others_key = {
                    (0,0,0,1,1):'leader',
                    (0,0,1,0,1):'bishop',
                    (0,1,0,0,1):'citizen',
                    }
        if ((len(history)) < 6) and (len(history) > 0):
            if history[-1][opp_index] == 'C':
                internal_memory[58][(len(history) + 1) - 2] = 0
            else:
                internal_memory[58][(len(history) + 1) - 2] = 1
        if (len(history) + 1) < 6:
            if bishop_key[(len(history))] == 0:
                return 'C'
            else:
                return 'D'
        else:
            if (len(history)) < 7:
                return 'C'
            others_pattern = others_key.get(tuple(internal_memory[58]))
            if others_pattern == None:
                if history[-1-1][opp_index] == 'D':
                    return 'D'
                else:
                    return history[-1][opp_index]
            else:
                if others_pattern == 'leader':
                    return 'C'
                else:
                    return 'D'
    elif strategy == 'Cult Bishop B':
        bishop_key = [0,0,1,0,1]
        others_key = {
                    (0,0,0,1,1):'leader',
                    (0,0,1,0,1):'bishop',
                    (0,1,0,0,1):'citizen',
                    }
        if ((len(history)) < 6) and (len(history) > 0):
            if history[-1][opp_index] == 'C':
                internal_memory[59][(len(history) + 1) - 2] = 0
            else:
                internal_memory[59][(len(history) + 1) - 2] = 1
        if (len(history) + 1) < 6:
            if bishop_key[(len(history))] == 0:
                return 'C'
            else:
                return 'D'
        else:
            if (len(history)) < 7:
                return 'C'
            others_pattern = others_key.get(tuple(internal_memory[59]))
            if others_pattern == None:
                if (history[-1-1][opp_index] == 'D') and (history[-1][opp_index] == 'D'):
                    return 'D'
                else:
                    return 'C'
            else:
                if others_pattern == 'leader':
                    return 'C'
                else:
                    return 'D'
    elif strategy == 'Cult Citizen A':
        bishop_key = [0,1,0,0,1]
        others_key = {
                    (0,0,0,1,1):'leader',
                    (0,0,1,0,1):'bishop',
                    (0,1,0,0,1):'citizen',
                    }
        if ((len(history)) < 6) and (len(history) > 0):
            if history[-1][opp_index] == 'C':
                internal_memory[60][(len(history) + 1) - 2] = 0
            else:
                internal_memory[60][(len(history) + 1) - 2] = 1
        if (len(history) + 1) < 6:
            if bishop_key[(len(history))] == 0:
                return 'C'
            else:
                return 'D'
        else:
            return 'C'
    elif strategy == 'Cult Citizen B':
        bishop_key = [0,1,0,0,1]
        others_key = {
                    (0,0,0,1,1):'leader',
                    (0,0,1,0,1):'bishop',
                    (0,1,0,0,1):'citizen',
                    }
        if ((len(history)) < 6) and (len(history) > 0):
            if history[-1][opp_index] == 'C':
                internal_memory[61][(len(history) + 1) - 2] = 0
            else:
                internal_memory[61][(len(history) + 1) - 2] = 1
        if (len(history) + 1) < 6:
            if bishop_key[(len(history))] == 0:
                return 'C'
            else:
                return 'D'
        else:
            if (len(history)) < 7:
                return 'C'
            others_pattern = others_key.get(tuple(internal_memory[61]))
            if others_pattern == None:
                return history[-1][opp_index]
            else:
                return 'C'
    elif strategy == 'Cult Citizen C':
        bishop_key = [0,1,0,0,1]
        others_key = {
                    (0,0,0,1,1):'leader',
                    (0,0,1,0,1):'bishop',
                    (0,1,0,0,1):'citizen',
                    }
        if ((len(history)) < 6) and (len(history) > 0):
            if history[-1][opp_index] == 'C':
                internal_memory[62][(len(history) + 1) - 2] = 0
            else:
                internal_memory[62][(len(history) + 1) - 2] = 1
        if (len(history) + 1) < 6:
            if bishop_key[(len(history))] == 0:
                return 'C'
            else:
                return 'D'
        else:
            if (len(history)) < 7:
                return 'C'
            others_pattern = others_key.get(tuple(internal_memory[62]))
            if others_pattern == None:
                return history[-1][self_index] if self_payoff_matrix[history[-1][self_index] + history[-1][opp_index]] >= self_avg_payoff_matrix else flip(history[-1][self_index])
            else:
                return 'C'
    elif strategy == 'Angry Tit For Tat':
        B = min(5, len(history))
        if len(history) > 0:
            inp = tuple([history[-i][opp_index] for i in range(B, 0, -1)])
            if (internal_memory[63] == 1) or (inp == tuple(["D"] * B)):
                if internal_memory[63] == 0:
                    internal_memory[63] = 1
                return 'D'
            else:
                return history[-1][opp_index]
        return 'C'
    elif strategy == 'RNN & RTRL':
        brain = internal_memory[65][0]
        if len(history) > 0:
            if history[-1][opp_index] == 'C':
                if history[-1][self_index] == 'C':
                    reward = 1
                else:
                    reward = 2
            else:
                if history[-1][self_index] == 'C':
                    reward = -2
                else:
                    reward = -1
            learning_rate = 0.1
            Error = [-((history[-1][self_index] == 'C') - brain["output old val"][0]) * reward, -((history[-1][self_index] == 'D') - brain["output old val"][1]) * reward]
            G = brain["G"]
            for learn in range(brain["n"]):
                derivative = 1 - (brain["n old val"][learn] * brain["n old val"][learn])
                for learn2 in range(4): brain["scalar trace wi"][learn][learn2] = derivative * (brain["input old val"][learn2] + brain["recurrent weight"][learn] * brain["old scalar trace wi"][learn][learn2])
                brain["scalar trace wr"][learn] = derivative * (brain["n old val"][learn] + brain["recurrent weight"][learn] * brain["old scalar trace wr"][learn])
                G[learn] = sum(Error[learn2] * brain["output weight"][learn2][learn] for learn2 in range(2))
                for learn2 in range(4): brain["input weight"][learn][learn2] -= learning_rate * G[learn] * brain["scalar trace wi"][learn][learn2]
                brain["recurrent weight"][learn] -= learning_rate * G[learn] * brain["scalar trace wr"][learn]
                brain["bias"][learn] -= learning_rate * G[learn] * derivative
            for learn in range(2):
                for learn2 in range(brain["n"]): brain["output weight"][learn][learn2] -= learning_rate * Error[learn] * brain["n old val"][learn2]
                brain["output bias"][learn] -= learning_rate * Error[learn]
            for i in range(len(brain["old scalar trace wi"])): brain["old scalar trace wi"][i] = brain["old scalar trace wi"][i][:]
            brain["old scalar trace wr"][:] = brain["old scalar trace wr"][:]
            for i in range(len(brain["old scalar trace wo"])): brain["old scalar trace wo"][i] = brain["old scalar trace wo"][i][:]
        self_share = (history[-1][self_index] == 'C') if len(history) > 0 else True
        self_steal = (history[-1][self_index] == 'D') if len(history) > 0 else False
        opp_share = (history[-1][opp_index] == 'C') if len(history) > 0 else True
        opp_steal = (history[-1][opp_index] == 'D') if len(history) > 0 else False
        inp = [opp_share, opp_steal, self_share, self_steal]
        brain["n real val"] = mth_mtx.matrix_function(
            mth_mtx.matrix_aritc(
                mth_mtx.matrix_aritc(
                    mth_mtx.matrix_list_nestlist(
                        inp, brain["input weight"], brain["n"], None, 'dot prod'
                    ), 
                    mth_mtx.matrix_aritc(
                            brain["n old real val"], brain["recurrent weight"], '*'
                    ), '+'
                ), brain["bias"], '+'
            ), (-1, 1), 'clamp'
        )
        brain["n val"] = mth_mtx.matrix_function(brain["n real val"], None, 'tanh')
        brain["output val"] = mth_mtx.matrix_function(
            mth_mtx.matrix_aritc(
                mth_mtx.matrix_list_nestlist(
                    brain["n val"], brain["output weight"], 2, None, 'dot prod'
                ), brain["bias"], '+'
            ), (-1, 1), 'clamp'
        )
        share_output = mth.exp(brain["output val"][0]) / (mth.exp(brain["output val"][0]) + mth.exp(brain["output val"][1]))
        steal_output = mth.exp(brain["output val"][1]) / (mth.exp(brain["output val"][0]) + mth.exp(brain["output val"][1]))
        brain["output old val"] = [share_output, steal_output]
        brain["n old real val"] = brain["n real val"][:]
        brain["n old val"] = brain["n val"][:]
        brain["input old val"] = inp[:]
        if share_output >= steal_output:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Dont Bite The Hand That Feeds You':
        internal_memory[66][1] *= 0.75
        if (history[-1][opp_index] == 'C') if len(history) > 0 else True:
            internal_memory[66][1] += 0.25
        internal_memory[66][0] *= (1 - internal_memory[66][1])
        if (internal_memory[66][0] >= 0.5) or ((history[-1][opp_index] == 'D') if len(history) > 0 else False):
            if internal_memory[66][0] < 0.5:
                internal_memory[66][0] = min((internal_memory[66][0] + 1), 5)
            return 'D'
        else:
            return 'C'
    elif strategy == 'Tri-Brain':
        tb_eyes = [('C', 'C') for _ in range(8)] + history
        tribrain_input_power = [0.125 * (i + 1) for i in range(8)]
        opp_inp = [(tb_eyes[-i][opp_index] == 'C') for i in range(8, 0, -1)] 
        self_inp = [(tb_eyes[-i][self_index] == 'C') for i in range(8, 0, -1)] 
        SH = (tb_eyes[-1][opp_index] == 'C')
        def clamp(x):
            return max(min(x,1),-1)
        def tribrains_norm(x):
            return (x - 0.5) * 2
        tribrain_logic = clamp(sum(((SH * (tribrain_input_power[i] * opp_inp[i])) - ((1-SH) * (tribrain_input_power[i] * (1 - opp_inp[i])))) + (tribrain_input_power[i] * (tribrains_norm(opp_inp[i]) + tribrains_norm(self_inp[i]))) for i in range(8)))
        tribrain_most = clamp(sum((tribrain_input_power[i] * opp_inp[i]) - (tribrain_input_power[i] * (1 - opp_inp[i])) for i in range(8)))
        tribrain_now = SH - (1-SH)
        tribrain_output = tribrain_logic + tribrain_most + tribrain_now
        if tribrain_output > 0:
            return 'C'
        elif tribrain_output == 0:
            return tb_eyes[-1][opp_index]
        else:
            return 'D'
    elif strategy == 'Racister':
        if (len(history)) < (tournament_avg_last_round / 2):
            if len(history) > 0:
                if history[-1][opp_index] == 'D':
                    internal_memory[67] = 1
            return 'C'
        else:
            if internal_memory[67] == 1:
                return history[-1][opp_index]
            else:
                return 'D'
    elif strategy == 'Karen':
        if internal_memory[68] == 0:
            if ((history[-1][opp_index] == 'C') if len(history) > 0 else True):
                if rnd.random() <= 0.1:
                    return 'D'
                else:
                    return 'C'
            else:
                internal_memory[68] = 4
                return rnd.choice(['C','D'])
        else:
            if history[-1][opp_index] == 'C':
                internal_memory[68] -= 1
                return rnd.choice(['C','D'])
            else:
                return 'D'
    elif strategy == 'Male / Man':
        if (len(history)) < ((50 / 200) * tournament_avg_last_round): 
            if (internal_memory[69] == 1) or ((history[-1][opp_index] == 'D') if len(history) > 0 else False):
                if internal_memory[69] == 0:
                    internal_memory[69] = 1
                return 'D'
            else:
                return 'C'
        elif (len(history)) < ((100 / 200) * tournament_avg_last_round): 
            if (internal_memory[69] == 1) or (history[-1][opp_index] == 'D'):
                if internal_memory[69] == 0:
                    internal_memory[69] = 1
                else:
                    if rnd.random() <= 0.1:
                        internal_memory[69] = 0
                return 'D'
            else:
                return 'C'
        elif (len(history)) < ((150 / 200) * tournament_avg_last_round): 
            return history[-1][opp_index]
        else: 
            if rnd.random() < 0.1:
                return 'C'
            else:
                return history[-1][opp_index]
    elif strategy == 'Female / Woman':
        if (len(history)) < ((50 / 200) * tournament_avg_last_round): 
            if ((history[-1][opp_index] == 'C') if len(history) > 0 else True):
                return 'C'
            else:
                return rnd.choice(['C','D'])
        elif (len(history)) < ((100 / 200) * tournament_avg_last_round): 
            if rnd.random() < 0.1:
                return 'D'
            else:
                return history[-1][opp_index]
        elif (len(history)) < ((150 / 200) * tournament_avg_last_round): 
            if rnd.random() < 0.1:
                return 'C'
            else:
                return history[-1][opp_index]
        else: 
            if (history[-1][opp_index] == 'D') and (history[-1-1][opp_index] == 'D'):
                return 'D'
            else:
                return 'C'
    elif strategy == 'Anatol':
        Ant_eyes = [('C', 'C') for _ in range(4)] + history
        self_inp = [(Ant_eyes[-i][self_index] == 'D') for i in range(4, 0, -1)] 
        opp_inp = [(Ant_eyes[-i][opp_index] == 'D') for i in range(4, 0, -1)] 
        t_d = sum(self_inp[i] + opp_inp[i] for i in range(4))
        self_t_d = sum(self_inp)
        if ((t_d >= 3) and (t_d <= 4)) and (self_t_d > 0):
            return 'C'
        else:
            return Ant_eyes[-1][opp_index]
    elif strategy == 'Good':
        brain = internal_memory[70][0]
        if len(history) > 0:
            if history[-1][opp_index] == 'C':
                if history[-1][self_index] == 'C':
                    reward_or_punisment = 1.25
                else:
                    reward_or_punisment = 1.5
            else:
                if history[-1][self_index] == 'C':
                    reward_or_punisment = 0.5
                else:
                    reward_or_punisment = 1.1
            brain[internal_memory[70][1]] = max(min(brain[internal_memory[70][1]] * reward_or_punisment,1),0)
        inp = tuple([history[-1][opp_index], history[-1][self_index]]) if len(history) > 0 else ('C', 'C')
        if brain.get(inp) == None:
            brain[inp] = 0.5
        internal_memory[70][1] = inp
        if rnd.random() <= brain[inp]:
            return 'C'
        else:
            return 'D'
    elif strategy == 'AQUA':
        if not len(history):
            return 'C'
        B = min(8, len(history))
        inp = [(history[-i][opp_index] == 'C') for i in range(B, 0, -1)]
        share_ratio = sum(inp) / B
        if rnd.random() <= share_ratio:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Majapahit':
        R = 5
        if (len(history)) < R:
            if len(history) > 0:
                if history[-1][opp_index] == 'D':
                    internal_memory[72][0] = 1
            return 'C'
        elif (len(history) + 1) == (R + 1):
            if internal_memory[72][0] == 1:
                return 'D'
            else:
                return 'C'
        else:
            if history[-1][opp_index] == 'D':
                internal_memory[72][1] = min(internal_memory[72][1] + 0.1, 5)
            if internal_memory[72][0] == 1:
                internal_memory[72][1] *= 0.9
                if rnd.random() < (internal_memory[72][1] + 0.1):
                    return 'D'
                else:
                    return history[-1][opp_index]
            else:
                return history[-1][opp_index]
    elif strategy == 'EGOist':
        ego_of_hate = max((((score[opp_index] - score[self_index]) + ((history[-1][opp_index] == 'D') if len(history) > 0 else False)) / 10),0)
        ego_of_wanting = 0.1 + (0.15 * ((history[-1][opp_index] == 'C') if len(history) > 0 else True))
        ego = ego_of_hate + ego_of_wanting
        if rnd.random() <= ego:
            return 'D'
        else:
            return 'C'
    elif strategy == 'The Calculated Mirror With A Fuse':
        if (len(history) + 1) > 2:
            if ((history[-1-1][opp_index] == 'D') and (history[-1][opp_index] == 'D')) or (internal_memory[73][1] > 0):
                if internal_memory[73][1] == 0:
                    internal_memory[73][1] = 3
                if internal_memory[73][1] < 3:
                    if history[-1][opp_index] != 'D':
                        internal_memory[73][1] -= 1
                else:
                    internal_memory[73][1] -= 1
                return 'D'
        if len(history) > 0:
            if history[-1][opp_index] == 'C':
                internal_memory[73][0] += 1
        if (len(history)) < 2:
            return 'C'
        A = internal_memory[73][0] / (len(history) + 1)
        if A >= 0.7:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Trust Ledger':
        if (len(history) + 1) > 3:
            inp = tuple([history[-i][opp_index] for i in range(3, 0, -1)]) 
            if (inp == ("C", "C", "C")) and (internal_memory[74] <= -6):
                internal_memory[74] = 0
        if len(history) > 0:
            if history[-1][opp_index] == 'C':
                internal_memory[74] += 2
            else:
                internal_memory[74] -= 3
        if internal_memory[74] > 0:
            internal_memory[74] -= 1
        elif internal_memory[74] < 0:
            internal_memory[74] += 1
        internal_memory[74] = max(min(internal_memory[74],10),-10)
        if internal_memory[74] >= 3:
            return 'C'
        elif internal_memory[74] <= -3:
            return 'D'
        else:
            if not len(history):
                return 'C'
            return history[-1][opp_index]
    elif strategy == 'Echo Adaptive':
        if ((history[-1-1][opp_index] == 'D') if (len(history) + 1) > 2 else False) and ((history[-1][opp_index] == 'D') if len(history) > 0 else False):
            return 'C'
        elif ((history[-1-2][opp_index] == 'D') if (len(history) + 1) > 3 else False) and ((history[-1-1][opp_index] == 'D') if (len(history) + 1) > 2 else False):
            return 'D'
        else:
            if not len(history):
                return 'C'
            return history[-1][opp_index]
    elif strategy == 'Threshold Punisher':
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[75] += 1
        else:
            return 'C'
        steal_ratio = internal_memory[75] / (len(history))
        if steal_ratio > 0.3:
            return 'D'
        else:
            return 'C'
    elif strategy == 'AMERICA! / Freedoom / USA':
        tm = internal_memory[77]
        opp = history[-1][opp_index] if len(history) > 0 else 'C'
        tm[0] += 0.1
        tm[0] = min(tm[0],1)
        exploit_chance = (
            (tm[0] * 0.6)
        )
        if opp == 'C':
            if rnd.random() < exploit_chance:
                tm[1] += 1
                return 'D'
            else:
                return 'C'
        else:
            return 'D'
    elif strategy == 'Inequality / SOVIET UNION! / USSR':
        tm = internal_memory[78]
        opp = history[-1][opp_index] if len(history) > 0 else 'C'
        if opp == 'C':
            tm[0] += 1
            tm[1] += 0.04
        else:
            tm[1] -= 0.08
        tm[1] = max(0,min(1,tm[1]))
        if rnd.random() < (0.05 * min(tm[0], 2)):
                return 'D'
        if rnd.random() < tm[1]:
            return 'C'
        if opp == 'D':
            return 'D'
        return 'C'
    elif strategy == 'Little Grudger':
        if len(history) > 0:
            internal_memory[79] *= 0.75
            if history[-1][opp_index] == 'C':
                internal_memory[79] += 0.25
        if rnd.random() <= internal_memory[79]:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Second Chance Attariq Version':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'D':
            if internal_memory[80][0] == 0:
                internal_memory[80][0] = 1
                internal_memory[80][1] = (len(history) + 1)
            elif internal_memory[80][0] == 1:
                internal_memory[80][0] = 2
                internal_memory[80][2] = (len(history) + 1)
            else:
                internal_memory[80][1] = internal_memory[80][2]
                internal_memory[80][2] = (len(history) + 1)
        if internal_memory[80][0] != 2:
            return 'C'
        else:
            A1 = max(min(1 - ((internal_memory[80][2] - internal_memory[80][1]) / 5),1),0)
            A2 = max(min(1 - (((len(history)) - internal_memory[80][2]) / 5),1),0)
            A = A1 * A2
            if rnd.random() <= A:
                return 'D'
            else:
                return 'C'
    elif strategy == 'Crabby Attariq Version':
        if not len(history):
            return 'D'
        elif (len(history) + 1) == 2:
            return 'C'
        elif (len(history) + 1) == 3:
            if history[-1][opp_index] == 'D':
                internal_memory[85][0] = 0
            else:
                internal_memory[85][0] = 1
            return 'C'
        if internal_memory[85][0] == 1:
            if (history[-1][opp_index] == 'D') and ((len(history) + 1) > 150):
                if internal_memory[85][1] == 0:
                    internal_memory[85][1] = 1
            if (len(history) + 1) > ((185 / 200) * tournament_avg_last_round):
                return 'D'
            if internal_memory[85][1] == 0:
                if (len(history) + 1) > ((100 / 200) * tournament_avg_last_round):
                    if ((len(history) + 1) % 5) == 0:
                        return 'D'
                    else:
                        return 'C'
                else:
                    if rnd.random() <= 0.2:
                        return 'C'
                    else:
                        return history[-1][opp_index]
            elif internal_memory[85][1] == 1:
                if rnd.random() <= 0.1:
                    if (history[-1][opp_index] == 'C'):
                        internal_memory[85][1] = 2
                        return 'C'
                return 'D'
            elif internal_memory[85][1] == 2:
                internal_memory[85][1] = 0
                return 'C'
        else:
            if rnd.random() <= 0.2:
                return 'C'
            else:
                return history[-1][opp_index]
    elif strategy == 'Cycler DCCCC':
        if not len(history):
            return 'D'
        elif (len(history) + 1) == 2:
            return 'C'
        elif (len(history) + 1) == 3:
            if history[-1][opp_index] == 'D':
                internal_memory[86] = 0
            else:
                internal_memory[86] = 1
            return 'C'
        if internal_memory[86] == 1:
            if (((len(history) + 1)-1) % 5) == 0:
                return 'D'
            else:
                if rnd.random() <= 0.2:
                    return 'C'
                else:
                    return history[-1][opp_index]
        else:
            return history[-1][opp_index]
    elif strategy == 'Odd-Even Go By Majority':
        if not len(history):
            return 'C'
        B = min(8, len(history))
        inp = [1 if history[-i][opp_index] == 'C' else -1 for i in range(B, 0, -1)]
        odt_sum_odd = sum(inp[(2*i)-2] for i in range(B // 2))
        odt_sum_even = sum(inp[(2*i)-1] for i in range(B - (B // 2)))
        if rnd.random() <= 0.5:
            odt_final_choice = odt_sum_even
        else:
            odt_final_choice = odt_sum_odd
        if odt_final_choice > 0:
            return 'C'
        elif odt_final_choice < 0:
            return 'D'
        else:
            return history[-1][opp_index]
    elif strategy == 'Desire':
        dsi_eyes = [('C', 'C') for _ in range(8)] + history
        input_power = [0.125 * (i + 1) for i in range(8)]
        D_input = [(dsi_eyes[-i][opp_index] == 'D') for i in range(8, 0, -1)] 
        C_input = [(dsi_eyes[-i][opp_index] == 'C') for i in range(8, 0, -1)] 
        A1 = sum(D_input[i] * input_power[i] for i in range(8))
        A2 = sum(C_input[i] * input_power[i] for i in range(8))
        if (score[self_index] - score[opp_index]) <= -10:
            A1 += -((score[self_index] - score[opp_index]) / 10)
            A2 += ((score[self_index] - score[opp_index]) / 10)
        elif (score[self_index] - score[opp_index]) >= 10:
            A1 += (score[self_index] - score[opp_index]) / 10
            A2 += -(score[self_index] - score[opp_index]) / 10
        A = A2 - A1
        if A > 0:
            return 'C'
        elif A == 0:
            return 'D' if rnd.random() <= 0.25 else 'C'
        else:
            return 'D'
    elif strategy == 'Lighty-Darky':
        if ((history[-1][opp_index] == 'D') if len(history) > 0 else False):
            internal_memory[87] += 1
        else:
            if rnd.random() <= 0.1:
                internal_memory[87] += 1
        if (len(history) + 1) > (tournament_avg_last_round / 2):
            if internal_memory[87] > 0:
                internal_memory[87] -= 1
                return 'D'
            else:
                return 'C'
        else:
            return 'C'
    elif strategy == 'Alcohol / Ethanol':
        if not len(history):
            return 'D'
        elif (len(history)) < (tournament_avg_last_round / 2):
            if history[-1][opp_index] == 'D':
                internal_memory[90] += 1
            return 'C'
        else:
            if history[-1][opp_index] == 'D':
                internal_memory[90] += 1
            if (internal_memory[90] - 1) > 0:
                internal_memory[90] -= 1
                return 'D'
            else:
                return 'C'
    elif strategy == 'Simple Exploiter':
        if (len(history)) < 2:
            return ('D', 'C')[len(history)]
        elif (len(history) + 1) == 3:
            if history[-1][opp_index] == 'D':
                internal_memory[92] = 1
                return 'C'
            else:
                internal_memory[92] = 0
                return 'D'
        else:
            if internal_memory[92] == 1:
                return history[-1][opp_index]
            else:
                return 'D'
    elif strategy == 'Smart Go By Majority':
        if (len(history)) < 8:
            if len(history) > 0:
                internal_memory[93][(len(history) + 1) - 2] = (history[-1][opp_index] == 'D')
            return 'C'
        def st_set_new_memory():
            for i in range(6):
                internal_memory[93][i] = internal_memory[93][i+1]
            internal_memory[93][6] = (history[-1][opp_index] == 'D')
        t_d = sum(internal_memory[93])
        A = 1 - (((t_d * t_d) - 1) / 49)
        if rnd.random() <= A:
            st_set_new_memory()
            return 'C'
        else:
            st_set_new_memory()
            return 'D'
    elif strategy == 'Mafia A':
        if (len(history)) < 3:
            return ('D', 'C', 'D')[len(history)]
        elif (len(history) + 1) == 4:
            inp = tuple(["D" if history[-i][opp_index] == 'D' else "C" for i in range(3, 0, -1)]) 
            if inp == ("D", "C", "D"):
                internal_memory[94] = 1
                return 'C'
            elif inp == ("C", "D", "C"):
                internal_memory[94] = 2
                return 'C'
            else:
                internal_memory[94] = 0
                return 'D'
        else:
            mafia_eyes = [('C', 'C') for _ in range(10)] + history
            opp_inp = tuple(["D" if mafia_eyes[-i][opp_index] == 'D' else "C" for i in range(3, 0, -1)]) 
            self_inp = tuple([mafia_eyes[-i][self_index] for i in range(3, 0, -1)]) 
            grudger_inp = [(mafia_eyes[-i][opp_index] == 'D') for i in range(10, 0, -1)] 
            if internal_memory[94] > 0:
                if internal_memory[94] == 2:
                    grudger_ratio = sum(grudger_inp) / 10
                    if (opp_inp, self_inp) in ((("C", "D", "C"), ("D", "C", "D")), (("D", "C", "D"), ("C", "D", "C"))):
                        return 'C'
                    else:
                        A = 1 - max(min(2 * (grudger_ratio - 0.5),1),0)
                        if rnd.random() <= (0.1 * A):
                            return 'C'
                        else:
                            return history[-1][opp_index]
                else:
                    return 'C'
            else:
                return 'D'
    elif strategy == 'Bluff':
        if (len(history)) < 4:
            return ('C', 'D', 'C', 'C')[(len(history) + 1)-1]
        else:
            if history[-1][opp_index] == 'D':
                return 'D'
            else:
                if (rnd.random() <= 0.5) and ((history[-1][self_index] == 'D') and (history[-1-1][opp_index] == 'D')):
                    return 'D'
                else:
                    return 'C'
    elif strategy == 'Mafia B':
        if (len(history)) < 3:
            return ('D', 'C', 'D')[len(history)]
        elif (len(history) + 1) == 4:
            inp = tuple(["D" if history[-i][opp_index] == 'D' else "C" for i in range(3, 0, -1)]) 
            if inp == ("D", "C", "D"):
                internal_memory[95] = 1
                return 'C'
            elif inp == ("C", "D", "C"):
                internal_memory[95] = 2
                return 'C'
            else:
                internal_memory[95] = 0
                return 'D'
        else:
            mafia_eyes = [('C', 'C') for _ in range(10)] + history
            opp_inp = tuple([mafia_eyes[-i][opp_index] for i in range(3, 0, -1)]) 
            self_inp = tuple([mafia_eyes[-i][self_index] for i in range(3, 0, -1)]) 
            grudger_inp = [(mafia_eyes[-i][opp_index] == 'D') for i in range(10, 0, -1)] 
            if internal_memory[95] > 0:
                if internal_memory[95] == 2:
                    grudger_ratio = sum(grudger_inp) / 10
                    if (opp_inp, self_inp) in ((("C", "D", "C"), ("D", "C", "D")), (("D", "C", "D"), ("C", "D", "C"))):
                        return 'C'
                    else:
                        A = 1 - max(min(2 * (grudger_ratio - 0.5),1),0)
                        if rnd.random() <= (0.1 * A):
                            return 'C'
                        else:
                            return history[-1][opp_index]
                else:
                    return 'C'
            else:
                if (len(history) + 1) % 2 == 0:
                    return 'D'
                else:
                    return history[-1][opp_index]
    elif strategy == 'Mafia C':
        if (len(history)) < 3:
            return ('D', 'C', 'D')[len(history)]
        elif (len(history) + 1) == 4:
            inp = tuple(["D" if history[-i][opp_index] == 'D' else "C" for i in range(3, 0, -1)]) 
            if inp == ("D", "C", "D"):
                internal_memory[96] = 1
                return 'C'
            elif inp == ("C", "D", "C"):
                internal_memory[96] = 2
                return 'C'
            else:
                internal_memory[96] = 0
                return 'D'
        else:
            mafia_eyes = [('C', 'C') for _ in range(10)] + history
            opp_inp = tuple([mafia_eyes[-i][opp_index] for i in range(3, 0, -1)]) 
            self_inp = tuple([mafia_eyes[-i][self_index] for i in range(3, 0, -1)]) 
            grudger_inp = [(mafia_eyes[-i][opp_index] == 'D') for i in range(10, 0, -1)] 
            if internal_memory[96] > 0:
                if internal_memory[96] == 2:
                    grudger_ratio = sum(grudger_inp) / 10
                    if (opp_inp, self_inp) in ((("C", "D", "C"), ("D", "C", "D")), (("D", "C", "D"), ("C", "D", "C"))):
                        return 'C'
                    else:
                        A = 1 - max(min(2 * (grudger_ratio - 0.5),1),0)
                        if rnd.random() <= (0.2 * A):
                            return 'C'
                        else:
                            return history[-1][opp_index]
                else:
                    return 'C'
            else:
                if history[-1][opp_index] == 'C':
                    if history[-1][self_index] == 'C':
                        return rnd.choice(['D','C'])
                    else:
                        return 'D'
                else:
                    if history[-1][self_index] == 'C':
                        return 'D'
                    else:
                        return rnd.choice(['D','C'])
    elif strategy == 'Mafia D':
        if (len(history)) < 3:
            return ('D', 'C', 'D')[len(history)]
        elif (len(history) + 1) == 4:
            inp = tuple(["D" if history[-i][opp_index] == 'D' else "C" for i in range(3, 0, -1)]) 
            if inp == ("D", "C", "D"):
                internal_memory[97] = 1
                return 'C'
            elif inp == ("C", "D", "C"):
                internal_memory[97] = 2
                return 'C'
            else:
                internal_memory[97] = 0
                return 'D'
        else:
            mafia_eyes = [('C', 'C') for _ in range(10)] + history
            opp_inp = tuple([mafia_eyes[-i][opp_index] for i in range(3, 0, -1)]) 
            self_inp = tuple([mafia_eyes[-i][self_index] for i in range(3, 0, -1)]) 
            grudger_inp = [(mafia_eyes[-i][opp_index] == 'D') for i in range(10, 0, -1)] 
            if internal_memory[97] > 0:
                if internal_memory[97] == 2:
                    grudger_ratio = sum(grudger_inp) / 10
                    if (opp_inp, self_inp) in ((("C", "D", "C"), ("D", "C", "D")), (("D", "C", "D"), ("C", "D", "C"))):
                        return 'C'
                    else:
                        A = 1 - max(min(2 * (grudger_ratio - 0.5),1),0)
                        if rnd.random() <= (0.2 * A):
                            return 'C'
                        else:
                            return history[-1][opp_index]
                else:
                    return 'C'
            else:
                if history[-1][opp_index] == 'C':
                    return rnd.choice(['D','C'])
                else:
                    return 'D'
    elif strategy == 'Stoicalist':
        M = 5
        a = 1 / M
        input_power = [a * (i+1) for i in range(M)]
        stc_eyes = [('C', 'C') for _ in range(M)] + history
        inp = [(stc_eyes[-i][opp_index] == 'D') for i in range(M, 0, -1)] 
        E = sum(input_power[i] * inp[i] for i in range(M)) / sum(input_power[j] for j in range(M))
        if E >= 0.5:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Sun Tzu Bot':
        if internal_memory[98] == 1:
            return 'D'
        else:
            stb_eyes = [('C', 'C') for _ in range(10)] + history
            inp = [(stb_eyes[-i][opp_index] == 'D') for i in range(10, 0, -1)] 
            grudger_ratio = sum(inp[i] for i in range(10)) / 10
            random_cc_ratio = sum(1 if (inp[i], inp[i+1]) == (0, 0) else 0 for i in range(9)) / 9
            random_cd_ratio = sum(1 if (inp[i], inp[i+1]) == (0, 1) else 0 for i in range(9)) / 9
            random_dc_ratio = sum(1 if (inp[i], inp[i+1]) == (1, 0) else 0 for i in range(9)) / 9
            random_dd_ratio = sum(1 if (inp[i], inp[i+1]) == (1, 1) else 0 for i in range(9)) / 9
            random_estimation = (1.5 - (abs(random_cc_ratio - 0.25) + abs(random_cd_ratio - 0.25) + abs(random_dc_ratio - 0.25) + abs(random_dd_ratio - 0.25))) / 1.5
            opp_inp = tuple([(stb_eyes[-i][opp_index] == 'D') for i in range(3, 0, -1)]) 
            self_inp = tuple([(stb_eyes[-i][self_index] == 'D') for i in range(3, 0, -1)]) 
            if random_estimation >= 0.75:
                internal_memory[98] = 1
                return 'D'
            else:
                if (opp_inp, self_inp) in (((1, 0, 1), (0, 1, 0)), ((0, 1, 0), (1, 0, 1))):
                    return 'C'
                else:
                    A = 1 - max(min(2 * (grudger_ratio - 0.5),1),0)
                    if rnd.random() <= (0.1 * A):
                        return 'C'
                    else:
                        return stb_eyes[-1][opp_index]
    elif strategy == 'Laws':
        brain = internal_memory[100][0]
        another_variable = internal_memory[100][1]
        another_variable[0] -= 1
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                another_variable[2] += 1
        A1 = 5
        A2 = 3
        laws_eyes = [('C', 'C') for _ in range(A1)] + history
        if len(brain) != 0:
            if laws_eyes[-1][opp_index] == 'C':
                if laws_eyes[-1][self_index] == 'C':
                    brain[another_variable[1]][1] += self_payoff_matrix['CC']
                else:
                    brain[another_variable[1]][1] += self_payoff_matrix['DC']
            else:
                if laws_eyes[-1][self_index] == 'C':
                    brain[another_variable[1]][1] -= 4 * (self_payoff_matrix['DD'] - self_payoff_matrix['CD'])
                else:
                    brain[another_variable[1]][1] += self_payoff_matrix['DD']
            if brain[another_variable[1]][1] <= 0 and another_variable[0] <= 0:
                new_output = "".join([rnd.choice(["C", "D", "T"]) for _ in range(A2)])
                brain[another_variable[1]] = [new_output,100]
            if brain[another_variable[1]][1] >= 200 and another_variable[0] <= 0:
                ST = (laws_eyes[-1][opp_index] == 'D')
                a = 0.25
                A = rnd.uniform(0.0, 1.5)
                if A <= (((another_variable[2] / (len(history) + 1)) * (1 - a)) + (ST * a)):
                    new_choice = "D"
                else:
                    if A > 1:
                        new_choice = "T"
                    else:
                        new_choice = "C"
                idx = rnd.randint(0,A2-1)
                new_output = brain[another_variable[1]][0][:idx] + new_choice + brain[another_variable[1]][0][idx+1:]
                brain[another_variable[1]] = [new_output,100]
        if another_variable[0] <= 0:
            inp = "".join([laws_eyes[-i][opp_index] for i in range(A1, 0, -1)]) 
            another_variable[0] = A2
            another_variable[1] = inp
            def similarity(new_key,old_key):
                new_key_vector_opp = [1 if new_key[S1] == "C" else -1 for S1 in range(A1)]
                old_key_vector_opp = [1 if old_key[S2] == "C" else -1 for S2 in range(A1)]
                similarity_val = 0
                for SRT in range(A1):
                    similarity_val += (new_key_vector_opp[SRT] * old_key_vector_opp[SRT])
                return ((similarity_val + A1) / 2) / (A1 * 2)
            if brain.get(inp) == None:
                if len(brain) != 0:
                    target_space = min(brain, key=lambda k: brain[k][1])
                    is_same = similarity(inp,target_space)
                    new_output = "".join([rnd.choice(["C", "D", "T"]) for _ in range(A2)])
                    if is_same >= rnd.random():
                        brain[target_space] = [new_output,100]
                        another_variable[1] = target_space
                    else:
                        brain[inp] = [new_output,100]
                        another_variable[1] = inp
                else:
                    new_output = "".join([rnd.choice(["C", "D", "T"]) for _ in range(A2)])
                    brain[inp] = [new_output,100]
                    another_variable[1] = inp
        idx = A2 - another_variable[0]
        idx = max(0, min(idx, (A2-1)))
        if brain[another_variable[1]][0][idx] == "C":
            return 'C'
        else:
            if brain[another_variable[1]][0][idx] == "T":
                return laws_eyes[-1][opp_index]
            else:
                return 'D'
    elif strategy == 'Matcher':
        if (len(history)) < 10:
            if not len(history):
                return 'C'
            return history[-1][opp_index]
        input_power = [0.1 * (i+1) for i in range(10)]
        cc_inp = [
                (1 if (history[-1 - ((9) - i)][opp_index] == 'C') else -1) if ((history[-1 - (((9) - i) + 1)][self_index] == 'C') and (history[-1 - (((9) - i) + 1)][opp_index] == 'C')) else 0 
            for i in range(1, 10)
            ]
        cd_inp = [
                (1 if (history[-1 - ((9) - i)][opp_index] == 'C') else -1) if ((history[-1 - (((9) - i) + 1)][self_index] == 'C') and (history[-1 - (((9) - i) + 1)][opp_index] == 'D')) else 0 
            for i in range(1, 10)
            ]
        dc_inp = [
                (1 if (history[-1 - ((9) - i)][opp_index] == 'C') else -1) if ((history[-1 - (((9) - i) + 1)][self_index] == 'D') and (history[-1 - (((9) - i) + 1)][opp_index] == 'C')) else 0 
            for i in range(1, 10)
            ]
        dd_inp = [
                (1 if (history[-1 - ((9) - i)][opp_index] == 'C') else -1) if ((history[-1 - (((9) - i) + 1)][self_index] == 'D') and (history[-1 - (((9) - i) + 1)][opp_index] == 'D')) else 0 
            for i in range(1, 10)
            ]
        matcher_dict = {
            'CC':sum(cc_inp[i] * input_power[i + 1] for i in range(9)),
            'CD':sum(cd_inp[i] * input_power[i + 1] for i in range(9)),
            'DC':sum(dc_inp[i] * input_power[i + 1] for i in range(9)),
            'DD':sum(dd_inp[i] * input_power[i + 1] for i in range(9))
        }
        matcher_cond = ('C' if history[-1][self_index] == 'C' else 'D') + ('C' if history[-1][opp_index] == 'C' else 'D')
        if matcher_dict[matcher_cond] >= 0:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Trust Score Sigmoid':
        if len(history) > 0:
            if history[-1][opp_index] == 'C':
                internal_memory[105] += 1
            else:
                internal_memory[105] -= 3
        def sigmoid(x):
            sigm = max(min(x,10),-10)
            return 1 / (1 + (mth.exp(-sigm)))
        A = sigmoid(internal_memory[105])
        if rnd.random() <= A:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Tit For Increasing Tat':
        tfit_eyes = [('C', 'C') for _ in range(5)] + history
        if internal_memory[106][1] > 0:
            internal_memory[106][1] -= 1
            return 'D'
        else:
            if tfit_eyes[-1][opp_index] == 'D':
                internal_memory[106][0] += 1
                internal_memory[106][1] = internal_memory[106][0] - 1
                return 'D'
            else:
                inp = tuple([tfit_eyes[-i][opp_index] for i in range(5, 0, -1)]) 
                if inp == ("C", "C", "C", "C", "C"):
                    internal_memory[106][0] = max(internal_memory[106][0] - 1, 0)
                return 'C'
    elif strategy == 'Freud':
        if len(history) > 0:
            if (history[-1][self_index] == 'C'):
                if (history[-1][opp_index] == 'D'):
                    internal_memory[107] -= 3
                else:
                    internal_memory[107] += 1
        id_wanting = internal_memory[107] + (score[self_index] - score[opp_index])
        ego_reality = (((history[-1][opp_index] == 'C') if len(history) > 0 else True) - 0.5) * 2
        super_ego_morality = (((history[-1][opp_index] == 'C') if len(history) > 0 else True) - 0.5) * 2 if rnd.random() > 0.2 else 1
        A = id_wanting + ego_reality + super_ego_morality
        if A == 0:
            if not len(history):
                return 'C'
            return history[-1][opp_index]
        elif A > 0:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Jung':
        jung_eyes = [('C', 'C') for _ in range(10)] + history
        if jung_eyes[-1][opp_index] == 'D':
            internal_memory[108] += 1
        else:
            internal_memory[108] -= 0.25
        persona = ((jung_eyes[-1][opp_index] == 'C') - 0.5) * 2 if rnd.random() > 0.2 else 1
        shadow = -internal_memory[108]
        def clamp(x):
            return max(min(x,1),0)
        input_power = [0.1 * (i+1) for i in range(10)]
        inp = [(jung_eyes[-i][opp_index] == 'C') for i in range(10, 0, -1)] 
        a = clamp(sum(input_power[i] * inp[i] for i in range(10)) / sum(input_power))
        A = (shadow * (1 - a)) + (persona * a)
        if A == 0:
            return jung_eyes[-1][opp_index]
        elif A > 0:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Skinner':
        def clamp(x):
            return max(min(x,1),-1)
        if len(history) > 0:
            if history[-1][opp_index] == 'C':
                if history[-1][self_index] == 'C':
                    internal_memory[109][0] += min(3 + max((score[self_index] - score[opp_index]), 0), 10)
                else:
                    internal_memory[109][1] += min(5 + max((score[self_index] - score[opp_index]), 0), 10)
            else:
                if history[-1][self_index] == 'C':
                    internal_memory[109][0] -= min(4 + max(-(score[self_index] - score[opp_index]), 0), 10)
                else:
                    internal_memory[109][1] -= min(1 + max(-(score[self_index] - score[opp_index]), 0), 10)
            internal_memory[109][0] = max(min(internal_memory[109][0], 100), -100)
            internal_memory[109][1] = max(min(internal_memory[109][1], 100), -100)
        E_c = (internal_memory[109][0] / (len(history))) if len(history) > 0 else 2.25
        E_d = (internal_memory[109][1] / (len(history))) if len(history) > 0 else 2.25
        A = E_c - E_d
        if A == 0:
            if not len(history):
                return 'C'
            return history[-1][opp_index]
        elif A > 0:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Rogers & Maslow':
        rogers_and_maslow_eyes = [('C', 'C') for _ in range(10)] + history
        input_power = [mth.exp(i+1) for i in range(10)]
        inp = [(rogers_and_maslow_eyes[-i][opp_index] == 'D') for i in range(10, 0, -1)] 
        D = sum(input_power[i] * inp[i] for i in range(10)) / sum(input_power)
        if rogers_and_maslow_eyes[-1][opp_index] == 'D':
            internal_memory[110] += 1
        else:
            internal_memory[110] *= D
        A = 1 - min(internal_memory[110], 1)
        if rnd.random() <= A:
            return 'C'
        else:
            if rnd.random() <= (1 - D):
                return 'C'
            else:
                return 'D'
    elif strategy == 'Frankl':
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[111] += 1
        frankl_choice = [0,0]
        frankl_choice[0] = (-1 if history[-1][opp_index] == 'D' else 1) if len(history) > 0 else 1
        Y_I_exist_qo = (abs((internal_memory[111] / len(history)) - 0.5) * 2) if len(history) > 0 else 0
        if rnd.random() < ((internal_memory[111] / len(history)) if len(history) > 0 else 0): 
            frankl_choice[1] = -1
        else:
            frankl_choice[1] = 1
        A = (frankl_choice[0] * (1 - Y_I_exist_qo)) + (frankl_choice[1] * Y_I_exist_qo)
        if A >= 0:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Teacher Pee While Standing, Student Pee While Running':
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[112] += 1
        a = 0.1
        ST = (history[-1][opp_index] == 'D') if len(history) > 0 else False
        A = ((internal_memory[112] / (len(history) + 1)) * (1 - a)) + (ST * a)
        if rnd.random() < A:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Javelin':
        javelin_eyes = [('C', 'C') for _ in range(5)] + history
        input_power = [0.2 * (i + 1) for i in range(5)]
        inp = [1 if javelin_eyes[-i][opp_index] == 'D' else -1 for i in range(5, 0, -1)] 
        A1 = sum(input_power[i] * inp[i] for i in range(5)) / sum(input_power)
        A2 = mth.tanh(A1 * 8)
        if rnd.random() <= A2:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Tip For Tap':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'C':
            internal_memory[113][1] += 1
        B = min(5, len(history))
        inp = [(history[-i][opp_index] == 'D') for i in range(B, 0, -1)]
        A = sum(inp) / B
        if A >= 0.5:
            return 'D'
        elif (history[-1][opp_index] == 'D') or (internal_memory[113][0] == 1):
            if internal_memory[113][0] == 0:
                internal_memory[113][0] = 1
                return 'C'
            else:
                internal_memory[113][0] = 0
                if history[-1][opp_index] == 'D':
                    return 'D'
                else:
                    p_c = (internal_memory[113][1] / (len(history) + 1))
                    if rnd.random() < (0.5 * p_c):
                        return 'C'
                    else:
                        return 'D'
        else:
            return 'C'
    elif strategy == 'Patterned Adaptive Fortress':
        confidence = (internal_memory[114][0] * 0.95) + (0.05 * ((history[-1][opp_index] == 'C') if len(history) > 0 else True))
        internal_memory[114][0] = confidence
        if (len(history)) < 5:
            return 'C'
        else:
            if len(history) >= (tournament_avg_last_round):
                if confidence < 0.9:
                    internal_memory[114][2] = 3
            if internal_memory[114][2] == 3:
                return 'D'
            if internal_memory[114][1] > 0:
                if history[-1][opp_index] == 'D':
                    internal_memory[114][2] += 1
                internal_memory[114][1] -= 1
                if internal_memory[114][1] > 0:
                    return 'C'
                else:
                    return history[-1][opp_index]
            if history[-1][opp_index] == 'C':
                return 'C'
            else:
                internal_memory[114][2] = 0
                internal_memory[114][1] = 4
                return 'D'
    elif strategy == 'Anti Noise Tit For Tat':
        if len(history) <= 1:
            return 'C'
        B = min(5, len(history))
        inp = [(history[-i][opp_index] == 'D') for i in range(B, 0, -1)]
        new_noise = sum(inp)
        if new_noise > 0:
            a = 0.25
            new_noise_prob = 1 - ((new_noise - 1) / (B - 1))
            internal_memory[115] = (internal_memory[115] * (1 - a)) + (new_noise_prob * a)
        if rnd.random() <= internal_memory[115]:
            return 'C'
        else:
            return history[-1][opp_index]
    elif strategy == 'Praedator':
        if (len(history)) < 5:
            if (len(history) + 1) == 5:
                if history[-1][opp_index] == 'D':
                    internal_memory[116][0] = 1
            return history[-1][opp_index] if len(history) > 0 else 'C'
        elif (len(history)) < 25:
            if history[-1][opp_index] == 'D':
                internal_memory[116][1] = 1
            return 'C'
        elif (len(history)) < 30:
            if (len(history) + 1) > 25:
                if history[-1][opp_index] == 'D':
                    internal_memory[116][2] = 1
            if (len(history) + 1) == 26:
                return 'D'
            else:
                return 'C'
        else:
            another_player_pattern = tuple(internal_memory[116])
            if another_player_pattern in ((0, 0, 1),(1, 0, 1),(0, 1, 1)):
                self_inp = [history[-i][self_index] for i in range(3, 0, -1)] 
                opp_inp = [history[-i][opp_index] for i in range(3, 0, -1)] 
                grudger_inp = [(history[-i][opp_index] == 'D') for i in range(10, 0, -1)] 
                grudger_ratio = sum(grudger_inp) / 10
                if ((self_inp == ["D", "C", "D"]) and (opp_inp == ["C", "D", "C"])) or ((opp_inp == ["D", "C", "D"]) and (self_inp == ["C", "D", "C"])):
                    return 'C'
                else:
                    A = 1 - max(min(2 * (grudger_ratio - 0.5),1),0)
                    if rnd.random() <= (0.2 * A):
                        return 'C'
                    else:
                        return history[-1][opp_index]
            elif another_player_pattern in ((1, 1, 1),(0, 0, 0)):
                return 'D'
            else:
                inp = [(history[-i][opp_index] == 'D') for i in range(5, 0, -1)] 
                t_d = sum(inp) / 5
                t_c = 1 - t_d
                ego = (score[self_index] - score[opp_index]) / 15
                if (t_c + max(ego, 0)) > (t_d - min(ego, 0)):
                    return 'C'
                else:
                    return 'D'
    elif strategy == 'Thresholded Trust Bot':
        thresholded_trust_bot_eyes = [('C', 'C') for _ in range(10)] + history
        a = 0.25
        internal_memory[117] += ((a if thresholded_trust_bot_eyes[-1][opp_index] == 'C' else -a) - ((1 - a) if thresholded_trust_bot_eyes[-1-1][opp_index] == 'C' else -(1 - a))) * 0.25
        internal_memory[117] = max(min(internal_memory[117], 1), 0)
        input_power = [mth.exp(i + 1) for i in range(10)]
        inp = [(thresholded_trust_bot_eyes[-i][opp_index] == 'C') for i in range(10, 0, -1)] 
        A = sum(input_power[i] * inp[i] for i in range(10)) / sum(input_power)
        if A == (1 - internal_memory[117]):
            return thresholded_trust_bot_eyes[-1][opp_index]
        elif A > (1 - internal_memory[117]):
            return 'C'
        else:
            return 'D'
    elif strategy == 'Forgetable Gradual':
        if not len(history):
            return 'C'
        if internal_memory[118][2] > 0:
            internal_memory[118][2] -= 1
            return 'C'
        if history[-1][opp_index] == 'D' or internal_memory[118][0] > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[118][1] += 1
                if internal_memory[118][0] == 0:
                    internal_memory[118][0] = internal_memory[118][1]
            internal_memory[118][0] -= 1
            if internal_memory[118][0] == 0:
                internal_memory[118][2] = 2
            return 'D'
        else:
            inp = tuple([history[-i][opp_index] for i in range(min(5, len(history)), 0, -1)])
            if inp == ("C", "C", "C", "C", "C"):
                internal_memory[118][1] -= 1
                internal_memory[118][1] = max(internal_memory[118][1],0)
            return 'C'
    elif strategy == 'MENACE':
        if len(history) > 0:
            rev = "C" if internal_memory[119][1] == "D" else "D"
            if history[-1][opp_index] == 'C':
                if history[-1][self_index] == 'C':
                    internal_memory[119][0][internal_memory[119][1]][0] += self_payoff_matrix['CC']
                    internal_memory[119][0][rev][1] -= self_payoff_matrix['CC'] / 2
                else:
                    internal_memory[119][0][internal_memory[119][1]][1] += self_payoff_matrix['DC']
                    internal_memory[119][0][rev][0] -= self_payoff_matrix['DC'] / 2
            else:
                if history[-1][self_index] == 'C':
                    internal_memory[119][0][internal_memory[119][1]][0] -= 4 * (self_payoff_matrix['DD'] - self_payoff_matrix['CD'])
                    internal_memory[119][0][rev][1] += 2 * (self_payoff_matrix['DD'] - self_payoff_matrix['CD'])
                else:
                    internal_memory[119][0][internal_memory[119][1]][1] += self_payoff_matrix['DD']
                    internal_memory[119][0][rev][0] -= self_payoff_matrix['DD'] / 2
            internal_memory[119][0][internal_memory[119][1]][0] = min(max(internal_memory[119][0][internal_memory[119][1]][0], (1 if internal_memory[119][0][internal_memory[119][1]][1] == 0 else 0)), 100)
            internal_memory[119][0][internal_memory[119][1]][1] = min(max(internal_memory[119][0][internal_memory[119][1]][1], (1 if internal_memory[119][0][internal_memory[119][1]][0] == 0 else 0)), 100)
            internal_memory[119][0][rev][0] = min(max(internal_memory[119][0][rev][0], (1 if internal_memory[119][0][rev][1] == 0 else 0)), 100)
            internal_memory[119][0][rev][1] = min(max(internal_memory[119][0][rev][1], (1 if internal_memory[119][0][rev][0] == 0 else 0)), 100)
        inp = history[-1][opp_index] if len(history) > 0 else 'C'
        act = internal_memory[119][0][inp][0] / (internal_memory[119][0][inp][0] + internal_memory[119][0][inp][1])
        internal_memory[119][1] = inp
        if rnd.random() <= act:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Forgiving Fate':
        internal_memory[120] *= 0.75
        if ((history[-1][opp_index] == 'C') if len(history) > 0 else True):
            internal_memory[120] += 0.25
        def C_function(x):
            return -(x - 1)
        ST = (history[-1][opp_index] == 'D') if len(history) > 0 else False
        A = 1 - (C_function(internal_memory[120]) * ST)
        if rnd.random() <= A:
            return 'C'
        else:
            return 'D'
    elif strategy == 'King Of 48 Laws Of Power':
        if not len(history):
            return 'C'
        inp = [history[i][opp_index] for i in range(len(history))]
        d_ratio = inp.count('D') / len(history)
        'law 04: always say less than necessary'
        if (score[self_index] - score[opp_index]) >= (self_payoff_matrix['DC'] * 2):
            next_chat[0] = 'I Am The King'
        elif (score[self_index] - score[opp_index]) <= -(self_payoff_matrix['DC'] * 2):
            'law 22: Use the Surrender Tactic: Transform Weakness into Power'
            next_chat[0] = 'I Am Lose'
        else:
            next_chat[0] = 'I Am Your Friend'
        if internal_memory[126] == 1:
            internal_memory[126] = 0
            return 'C'
        elif internal_memory[126] > 1:
            internal_memory[126] -= (history[-1][opp_index] == 'C')
            return 'D'
        'law 01: never outshine the master'
        'law 41: Avoid Stepping into a Great Man’s Shoes'
        if d_ratio <= 0.1 and history[-1][self_index] == 'D':
            internal_memory[126] = 1
            return 'C'
        'law 02: never put too much trust in friends, learn how to use enemy'
        'law 33: Discover Each Man’s Thumbscrew'
        A = frequency(history[-1][self_index], 'C', history, opp_index)
        if A != None and d_ratio > 0.1:
            if rnd.random() <= A:
                return 'D'
        'law 03: conceal your intentions'
        'law 05: So Much Depends on Reputation — Guard It with Your Life'
        if rnd.random() <= d_ratio and history[-1][opp_index] == 'D':
            return 'D'
        'law 06: court attention at all costs'
        'law 07: get others do the work for you'
        'law 08: make other people come to you, use bait if necessary'
        'law 27: Play on People’s Need to Believe to Create a Cultlike Following'
        if rnd.random() <= (2.5 * max(0.1 - d_ratio, 0)):
            return 'C'
        'law 09: win through your actions, not through argument'
        'law 10: Infection: Avoid the Unhappy and Unlucky'
        'law 15: Crush Your Enemy Totally'
        'law 18: Do Not Build Fortresses to Protect Yourself – Isolation is Dangerous'
        if d_ratio >= 0.5 and inp[-3:] == ['D'] * 3:
            internal_memory[126] = 3
            return 'D'
        'law 11: Learn to Keep People Dependent on You'
        'law 12: Use Selective Honesty and Generosity to Disarm Your Victim'
        if ('C', 'C') not in history[-5:] and rnd.random() <= max(0.1 - d_ratio, 0):
            internal_memory[126] = 1
            return 'C'
        'law 13: When Asking for Help, Appeal to People’s Self-Interest, Never to Their Mercy or Gratitude'
        'law 14: Pose as a Friend, Work as a Spy'
        'law 16: Use Absence to Increase Respect and Honor'
        'law 17: Keep Others in Suspended Terror: Cultivate an Air of Unpredictability'
        if rnd.random() <= max(d_ratio - 0.1, 0):
            return 'D'
        'law 19: Know Who You Are Dealing with – Do Not Offend the Wrong Person'
        'law 20: Do Not Commit to Anyone'
        'law 21: Play a Sucker to Catch a Sucker – Seem Dumber Than Your Mark'
        'law 32: Play to People’s Fantasies'
        'law 46: Never Appear Too Perfect'
        if rnd.random() <= (0.1 * (1 - d_ratio)):
            return 'C'
        'law 23: Concentrate Your Forces'
        'law 24: Play the Perfect Courtier'
        'law 25: Re-create Yourself'
        'law 26: Keep Your Hands Clean'
        'law 28: Enter Action with Boldness'
        'law 29: Plan All the Way to the End'
        'law 30: Make Your Accomplishments Seem Effortless'
        'law 31: Control the Options: Get Others to Play with the Cards You Deal'
        'law 34: Be Royal in Your Own Fashion: Act Like a King to Be Treated Like One'
        'law 35: Master the Art of Timing'
        'law 36: Disdain Things You Cannot Have: Ignoring Them is the Best Revenge'
        'law 37: Create Compelling Spectacles'
        'law 39: Stir Up Waters to Catch Fish'
        'law 40: Despise the Free Lunch'
        if history[-1][opp_index] == 'C' and d_ratio > 0.5:
            return 'D'
        'law 42: Strike the Shepherd and the Sheep Will Scatter'
        'law 43: Work on the Hearts and Minds of Others'
        'law 45: Preach the Need for Change, But Never Reform Too Much at Once'
        'law 47: In Victory, Learn When to Stop'
        'law 48: Assume Formlessness'
        'law 38: Think As You Like, But Act Like Others'
        'law 44: Disarm and Infuriate with the Mirror Effect'
        return history[-1][opp_index] if rnd.random() <= 0.1 else 'C'
    elif strategy == 'Pavolovo':
        if (len(history)) < 2:
            if not len(history):
                internal_memory[127][0] += 1
                internal_memory[127][2] = 1
            return ('D','C')[(len(history) + 1)-1]
        pavolovo_eyes = [('C', 'C') for _ in range(5)] + history
        if internal_memory[127][2] == 2:
            if pavolovo_eyes[-1][opp_index] == pavolovo_eyes[-1][self_index]:
                if (pavolovo_eyes[-1][opp_index] == 'D') and (pavolovo_eyes[-1-1][opp_index] == 'D'):
                    return 'D'
                return 'C'
            else:
                return 'D'
        if (pavolovo_eyes[-1-1][self_index] == 'D') or (internal_memory[127][2] == 1):
            inp = [(pavolovo_eyes[-i][opp_index] == 'D') for i in range(5, 0, -1)] 
            if sum(inp) == 5:
                internal_memory[127][2] = 2
                return 'D'
            if pavolovo_eyes[-1][opp_index] == 'D':
                internal_memory[127][1] += 1
                internal_memory[127][2] = 1
                return 'C'
            else:
                internal_memory[127][2] = 0
        p_retalitory = internal_memory[127][1] / internal_memory[127][0]
        if pavolovo_eyes[-1][opp_index] == pavolovo_eyes[-1][self_index]:
            if (pavolovo_eyes[-1][opp_index] == 'D') and (pavolovo_eyes[-1-1][opp_index] == 'D'):
                return 'D'
            else:
                if rnd.random() <= (0.1 * (1 - p_retalitory)):
                    internal_memory[127][0] += 1
                    return 'D'
                else:
                    return 'C'
        else:
            internal_memory[127][0] += 1
            return 'D'
    elif strategy == 'Decay Given':
        decay_given_eyes = [('C', 'C') for _ in range(10)] + history
        input_power = [0.1 * (i + 1) for i in range(10)]
        opp_inp = [(decay_given_eyes[-i][opp_index] == 'C') for i in range(10, 0, -1)] 
        self_inp = [(decay_given_eyes[-i][self_index] == 'C') for i in range(10, 0, -1)] 
        p_c_decay_given_c = sum(input_power[i] * opp_inp[i] * self_inp[i] for i in range(10)) / (sum(input_power[j] * self_inp[j] for j in range(10)) + 1e-9)
        p_c_decay_given_d = sum(input_power[i] * opp_inp[i] * (1 - self_inp[i]) for i in range(10)) / (sum(input_power[j] * (1 - self_inp[j]) for j in range(10)) + 1e-9)
        p_d_decay_given_c = sum(input_power[i] * (1 - opp_inp[i]) * self_inp[i] for i in range(10)) / (sum(input_power[j] * self_inp[j] for j in range(10)) + 1e-9)
        p_d_decay_given_d = sum(input_power[i] * (1 - opp_inp[i]) * (1 - self_inp[i]) for i in range(10)) / (sum(input_power[j] * (1 - self_inp[j]) for j in range(10)) + 1e-9)
        if decay_given_eyes[-1][self_index] == 'C':
            A = p_c_decay_given_c / (p_c_decay_given_c + p_d_decay_given_c + 1e-9)
            if rnd.random() <= A:
                return 'C'
            else:
                return 'D'
        else:
            A = p_c_decay_given_d / (p_c_decay_given_d + p_d_decay_given_d + 1e-9)
            if rnd.random() <= A:
                return 'D'
            else:
                return 'C'
    elif strategy == 'Eric':
        eric_eyes = [('C', 'C')] + history
        if (len(history)) < 5:
            return eric_eyes[-1][opp_index]
        if internal_memory[130] == 1:
            return eric_eyes[-1][opp_index]
        inp = [(eric_eyes[-i][opp_index] == 'D') for i in range(6, 0, -1)] 
        if (eric_eyes[-1][opp_index] == 'D') and (eric_eyes[-1-1][opp_index] == 'D'):
            return 'D'
        else:
            if sum(inp) >= 3:
                internal_memory[130] = 1
                return 'D'
            else:
                return 'C'
    elif strategy == 'Go By Minority':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'C':
            internal_memory[132][0] += 1
        else:
            internal_memory[132][1] += 1
        t_c = internal_memory[132][0]
        t_d = internal_memory[132][1]
        if t_c < t_d:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Score Equalizer':
        E_c = (abs((score[self_index] + self_payoff_matrix['CC']) - (score[opp_index] + self_payoff_matrix['CC'])) + abs((score[self_index] + self_payoff_matrix['CD']) - (score[opp_index] + self_payoff_matrix['DC']))) / 2
        E_d = (abs((score[self_index] + self_payoff_matrix['DC']) - (score[opp_index] + self_payoff_matrix['CD'])) + abs((score[self_index] + self_payoff_matrix['DD']) - (score[opp_index] + self_payoff_matrix['DD']))) / 2
        if E_c <= E_d:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Boltzmann Brain':
        def clamp(x):
            return max(min(x,1),0)
        brain = internal_memory[137][0]
        if len(history) > 0:
            M = 0.3
            if history[-1][opp_index] == 'C':
                boltzman_cond = 'win'
            else:
                boltzman_cond = 'lose'
            boltzman_update_list = []
            for i in range(len(brain)):
                if internal_memory[137][1][i] == (history[-1][self_index]):
                    boltzman_update_list.append(i)
            if boltzman_cond == 'win':
                for i in boltzman_update_list:
                    brain.append(
                        {
                        ("C", "C", "T", "F"):clamp(brain[i][("C", "C", "T", "F")] + rnd.uniform(-M,M)),
                        ("D", "C", "T", "F"):clamp(brain[i][("D", "C", "T", "F")] + rnd.uniform(-M,M)),
                        ("C", "D", "T", "F"):clamp(brain[i][("C", "D", "T", "F")] + rnd.uniform(-M,M)),
                        ("D", "D", "T", "F"):clamp(brain[i][("D", "D", "T", "F")] + rnd.uniform(-M,M)),
                        ("C", "C", "F", "F"):clamp(brain[i][("C", "C", "F", "F")] + rnd.uniform(-M,M)),
                        ("D", "C", "F", "F"):clamp(brain[i][("D", "C", "F", "F")] + rnd.uniform(-M,M)),
                        ("C", "D", "F", "F"):clamp(brain[i][("C", "D", "F", "F")] + rnd.uniform(-M,M)),
                        ("D", "D", "F", "F"):clamp(brain[i][("D", "D", "F", "F")] + rnd.uniform(-M,M)),
                        ("C", "C", "F", "T"):clamp(brain[i][("C", "C", "F", "T")] + rnd.uniform(-M,M)),
                        ("D", "C", "F", "T"):clamp(brain[i][("D", "C", "F", "T")] + rnd.uniform(-M,M)),
                        ("C", "D", "F", "T"):clamp(brain[i][("C", "D", "F", "T")] + rnd.uniform(-M,M)),
                        ("D", "D", "F", "T"):clamp(brain[i][("D", "D", "F", "T")] + rnd.uniform(-M,M))
                        }
                    )
                    internal_memory[137][1].append("C")
            else:
                for i in range(len(boltzman_update_list)):
                    brain.pop(boltzman_update_list[len(boltzman_update_list) - (i + 1)])
                    internal_memory[137][1].pop(boltzman_update_list[len(boltzman_update_list) - (i + 1)])
        if len(brain) > 32:
            while len(brain) > 32:
                A = rnd.randint(1,len(brain)) - 1
                brain.pop(A)
                internal_memory[137][1].pop(A)
        if len(brain) < 1:
            brain.append(
                {
                    ("C", "C", "T", "F"):1,
                    ("D", "C", "T", "F"):1,
                    ("C", "D", "T", "F"):1,
                    ("D", "D", "T", "F"):1,
                    ("C", "C", "F", "F"):1,
                    ("D", "C", "F", "F"):0,
                    ("C", "D", "F", "F"):1,
                    ("D", "D", "F", "F"):0,
                    ("C", "C", "F", "T"):0,
                    ("D", "C", "F", "T"):0,
                    ("C", "D", "F", "T"):0,
                    ("D", "D", "F", "T"):0
                }
            )
            internal_memory[137][1].append("C")
        opp = history[-1][opp_index] if len(history) > 0 else 'C'
        self = history[-1][self_index] if len(history) > 0 else 'C'
        boltzman_win = "T" if score[self_index] > score[opp_index] else "F"
        boltzman_lose = "T" if score[opp_index] > score[self_index] else "F"
        boltzmann_mini_strat_choice = []
        for i in range(len(brain)):
            if rnd.random() <= brain[i][(opp, self, boltzman_win, boltzman_lose)]:
                boltzmann_mini_strat_choice.append(1)
                internal_memory[137][1][i] = "C"
            else:
                boltzmann_mini_strat_choice.append(-1)
                internal_memory[137][1][i] = "D"
        A = sum(boltzmann_mini_strat_choice)
        if A <= 0:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Smooth Tit For Tat':
        a = 0.25
        internal_memory[138] *= (1 - a)
        if ((history[-1][opp_index] == 'D') if len(history) > 0 else False):
            internal_memory[138] += a
        if rnd.random() <= internal_memory[138]:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Konflikt Green':
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[99] += 1
        if rnd.random() <= (internal_memory[99] / (len(history) + 1)):
            return 'D'
        else:
            return 'C' if rnd.random() <= 0.25 else (history[-1][opp_index] if len(history) > 0 else 'C')
    elif strategy == 'Konflikt Blue':
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[133] += 1
        if rnd.random() <= ((internal_memory[133] / (len(history) + 1)) + 0.1):
            return 'D'
        else:
            return 'C' if rnd.random() <= 0.125 else (history[-1][opp_index] if len(history) > 0 else 'C')
    elif strategy == 'Fibonacci':
        if not len(history):
            internal_memory[144][3] = internal_memory[144][2]
        if internal_memory[144][3] == 0:
            internal_memory[144][0] = flip(internal_memory[144][0])
            internal_memory[144][3] = internal_memory[144][1] + internal_memory[144][2]
            internal_memory[144][1], internal_memory[144][2] = internal_memory[144][2], internal_memory[144][3]
        internal_memory[144][3] -= 1
        return internal_memory[144][0]
    elif strategy == 'Contrite Grudger':
        if internal_memory[147][1] == 0:
            if ((history[-1][self_index] == 'D') if len(history) > 0 else False) and ((history[-2][opp_index] == 'C') if len(history) > 0 else True):
                internal_memory[147][0] += 1
            if ((history[-1][opp_index] == 'D') if len(history) > 0 else False):
                if internal_memory[147][0] > 0:
                    internal_memory[147][0] -= 1
                    return 'C'
                else:
                    internal_memory[147][1] = 1
                    return 'D'
            else:
                return 'C'
        else:
            return 'D'
    elif strategy == 'Passive Tit For Tat':
        if not len(history):
            return 'C'
        elif history[-1][self_index] == 'D':
            return 'C'
        elif (history[-1][opp_index] == 'C') and ((history[-1-1][opp_index] == 'C') if (len(history) + 1) > 2 else True):
            return 'C'
        else:
            return 'D'
    elif strategy == 'Beta Tester':
        beta_tester_random_param = 0.2
        if rnd.random() <= beta_tester_random_param:
            internal_memory[29] = rnd.randint(0, 6)
        if internal_memory[29] == 0:
            return 'C'
        elif internal_memory[29] == 1:
            return 'D'
        elif internal_memory[29] == 2:
            return rnd_prob(0.5)
        elif internal_memory[29] == 3:
            if not len(history):
                return 'D'
            return 'D' if history[-1][self_index] == 'C' else 'C'
        elif internal_memory[29] == 4:
            if not len(history):
                return 'C'
            return history[-1][self_index] if self_payoff_matrix[history[-1][self_index] + history[-1][opp_index]] >= self_avg_payoff_matrix else flip(history[-1][self_index])
        elif internal_memory[29] == 5:
            if not len(history):
                return 'D'
            return 'D' if history[-1][opp_index] == 'C' else 'C'
        else:
            if not len(history):
                return 'C'
            return history[-1][opp_index]
    elif strategy == 'Buzzer':
        if len(history) <= (tournament_avg_last_round // 5):
            next_chat[0] = 'BUZZER!!!'
            if len(history):
                if history[-1][opp_index] == 'D' or chats[opp_index] == 'BUZZER!!!':
                    internal_memory[173] = True
            return 'C'
        else:
            if internal_memory[173]:
                return 'C'
            return 'D'
    elif strategy == 'Man In The Middle':
        if (len(history)) < ((190 / 200) * tournament_avg_last_round):
            if not len(history):
                return 'C'
            return 'C' if rnd.random() <= 0.1 else history[-1][opp_index]
        else:
            return 'D'
    elif strategy == 'DDoS':
        if (len(history)) < (tournament_avg_last_round / 2):
            if not len(history):
                return 'C'
            return 'C' if rnd.random() <= 0.1 else history[-1][opp_index]
        inp = [history[-i][opp_index] for i in range(1, 10 + 1)] 
        if "D" in inp:
            for i in range(5):
                DDoS_n = (i + 1)
                is_cycle = True
                DDoS_list = inp[:(DDoS_n + 1)]
                DDoS_index = len(inp) % DDoS_n
                for j in range(len(inp)):
                    if inp[j] != DDoS_list[j % DDoS_n]:
                        is_cycle = False
                if is_cycle:
                    if DDoS_list[DDoS_index + 1] == "D":
                        return 'D'
                    else:
                        return rnd_prob(0.5)
        return 'C' if rnd.random() <= 0.1 else history[-1][opp_index]
    elif strategy == 'Social Engineering':
        if (len(history)) < 5:
            return ('D', 'C', 'D', 'D', 'C')[len(history)]
        elif (len(history) + 1) == 6:
            inp = [history[-i][opp_index] for i in range(5, 0, -1)] 
            if tuple(inp) == ('D', 'C', 'D', 'D', 'C'):
                internal_memory[174] = 1
            return 'C'
        else:
            if internal_memory[174] == 1:
                return 'C'
            else:
                return history[-1][opp_index]
    elif strategy == 'Zero Day':
        A = 1
        if (len(history)) < (A + 1):
            if not len(history):
                return 'C'
            return history[-1][opp_index]
        elif (len(history)) < ((190 / 200) * tournament_avg_last_round):
            inp = ''
            for i in range(A):
                if history[-1 - (A - i)][opp_index] == 'C': 
                    inp += 'C'
                else:
                    inp += 'D'
                if history[-1 - (A - i)][self_index] == 'C': 
                    inp += 'C'
                else:
                    inp += 'D'
            if internal_memory[175][0].get(inp) == None:
                internal_memory[175][0][inp] = [(history[-1][opp_index] == 'C'), 1]
            else:
                internal_memory[175][0][inp][0] += (history[-1][opp_index] == 'C')
                internal_memory[175][0][inp][1] += 1
            cond = ''
            for i in range(A):
                if history[-1 - (A - i)][opp_index] == 'C': 
                    cond += 'C'
                else:
                    cond += 'D'
                if history[-1 - (A - i)][self_index] == 'C': 
                    cond += 'C'
                else:
                    cond += 'D'
            if cond != 'CC':
                B = 1 - (internal_memory[175][0][cond][0] / internal_memory[175][0][cond][1] if internal_memory[175][0].get(cond) != None else 0.5)
            else:
                B = internal_memory[175][0]['CC'][0] / internal_memory[175][0]['CC'][1] if internal_memory[175][0].get('CC') != None else 0.5
            return history[-1][opp_index] if rnd.random() <= B else 'D'
        else:
            return 'D'
    elif strategy == 'Eliza / Therapist':
        if ((history[-2][opp_index] == 'C') if (len(history) + 1) > 2 else True) and ((history[-1][self_index] == 'D') if len(history) > 0 else False):
            internal_memory[177][0] += 1
        opp_chat = chats[opp_index].lower().strip().rstrip('?.!')
        if opp_chat == '':
            opp_chat = 'we are friends'
            if len(history) > 0:
                opp_chat = 'we are friends' if history[-1][opp_index] == 'C' else 'i hate you'
        opp_chat = opp_chat.replace('were', 'we are')
        keyword_map = internal_memory[177][1]
        ganti_kata = {
            ' me ': ' you ',
            ' you ': ' me ',
            ' my ': ' your ',
            ' your ': ' my ',
        }
        pola = re.compile("|".join(sorted(map(re.escape, ganti_kata.keys()), key=len, reverse=True)))
        the_key_word = None
        for kw in sorted(keyword_map.keys(), key=len, reverse=True):
            if kw in opp_chat:
                the_key_word = kw
                break
        if the_key_word:
            text_template = rnd.choice(keyword_map[the_key_word])
            if '[REPLACE]' in text_template:
                A = (' ' + opp_chat + ' ').split(the_key_word, 1)[1]
                hasil = pola.sub(lambda m: ganti_kata[m.group(0)], A)
                next_chat[0] = f"{text_template.replace('[REPLACE]', hasil[1:-1])}"
            else:
                next_chat[0] = f"{text_template}"
        else:
            next_chat[0] = f"{rnd.choice(['What do you think about that?', 'Could you tell me more about it?', 'Hmm, interesting... go on?', 'How do you feel about that?'])}"
        if internal_memory[177][0] > 0:
            internal_memory[177][0] -= 1
            return 'C'
        else:
            if not len(history):
                return 'C'
            return history[-1][opp_index] if rnd.random() > 0.1 else 'C'
    elif strategy == 'Verity':
        def clamp(x):
            return max(min(x, 1), 0)
        verity_eyes = [('C', 'C') for _ in range(10)] + history
        if internal_memory[178][2] > 0:
            internal_memory[178][2] -= 1
            return 'C'
        if verity_eyes[-1][opp_index] == 'D':
            internal_memory[178][0] -= 30
            if internal_memory[178][1] == 1:
                internal_memory[178][1] = 2
        else:
            internal_memory[178][0] += 5
        verity_cond = (verity_eyes[-1-1][opp_index], verity_eyes[-1-1][self_index])
        internal_memory[178][6][verity_cond][0] += (verity_eyes[-1][opp_index] == 'C')
        internal_memory[178][6][verity_cond][1] += 1
        if (internal_memory[178][0] < 30) or internal_memory[178][3]:
            if not internal_memory[178][3]:
                internal_memory[178][3] = True
                internal_memory[178][4] = (len(history) + 1)
                internal_memory[178][5] = 20
            if internal_memory[178][1] < 3:
                internal_memory[178][1] = 3
            if (internal_memory[178][1] == 4) and (internal_memory[178][0] >= 30):
                internal_memory[178][3] = False
                internal_memory[178][2] = 1
                internal_memory[178][1] = 1
                return 'C'
            elif (((len(history) + 1) - internal_memory[178][4]) / internal_memory[178][5]) >= 0.75:
                internal_memory[178][1] = 4
        if internal_memory[178][1] == 1:
            return 'C'
        elif internal_memory[178][1] == 2:
            cyclic_inp = [verity_eyes[-i][opp_index] for i in range(1, 10 + 1)] 
            is_cycle = False
            if "D" in cyclic_inp:
                if cycle_detec(history[-10:], opp_index, min_ = 1, max_ = 5):
                    is_cycle = True
            if (verity_eyes[-1][opp_index] == 'C') and (not is_cycle):
                internal_memory[178][2] = 1
                internal_memory[178][1] = 1
                return 'C'
            return 'D'
        elif internal_memory[178][1] == 3:
            mutual_cooperate_prob = ((internal_memory[178][6][("C", "C")][0] + 1) / (internal_memory[178][6][("C", "C")][1] + 2))
            exploitation_prob = ((internal_memory[178][6][("C", "D")][0] + 1) / (internal_memory[178][6][("C", "D")][1] + 2))
            exploited_prob = ((internal_memory[178][6][("D", "C")][0] + 1) / (internal_memory[178][6][("D", "C")][1] + 2))
            mutual_defect_prob = ((internal_memory[178][6][("D", "D")][0] + 1) / (internal_memory[178][6][("D", "D")][1] + 2))
            verity_category = [
                [ 
                    1,
                    0,
                    1,
                    0
                ],
                [ 
                    1,
                    0,
                    0,
                    1
                ],
                [ 
                    0,
                    0,
                    1,
                    1
                ]
            ]
            verity_enemy_differents = [0 for _ in range(len(verity_category))]
            for i in range(len(verity_category)):
                verity_enemy_differents[i] = (
                    abs(mutual_cooperate_prob - verity_category[i][0]) +
                    abs(exploitation_prob - verity_category[i][1]) +
                    abs(exploited_prob - verity_category[i][2]) +
                    abs(mutual_defect_prob - verity_category[i][3])
                ) / 4
            if min(verity_enemy_differents) <= 0.25:
                most_similar = [['Tit For Tat', 'Pavlov', 'Alternator'][j] for j in range(len(verity_enemy_differents)) if verity_enemy_differents[j] == min(verity_enemy_differents)]
                enemy_category = rnd.choice(most_similar)
                if enemy_category == 'Tit For Tat':
                    return 'C'
                elif enemy_category == 'Pavlov':
                    return ['D', 'D', 'C'][(len(history) + 1) % 3]
                else:
                    return 'D' if verity_eyes[-1][opp_index] == 'C' else rnd_prob(0.5)
            if (verity_eyes[-1-1][opp_index], verity_eyes[-1][opp_index]) == ('D', 'D'):
                return 'D'
            if verity_eyes[-1][opp_index] == verity_eyes[-1][self_index]:
                return 'D' if rnd.random() <= exploitation_prob else 'C'
            else:
                return 'D'
        else:
            return 'D'
    elif strategy == 'Continuous Q-Learner Max':
        brain = internal_memory[179]
        reward = self_avg_payoff_matrix
        if len(history) > 0:
            learning_rate = 0.1
            brain[0] = ((1 - learning_rate) * brain[0]) + (learning_rate * (history[-1][opp_index] == 'C'))
            brain[1] = ((1 - learning_rate) * brain[1]) + (learning_rate * (history[-1][opp_index] == 'D'))
            reward = self_payoff_matrix[history[-1][self_index] + history[-1][opp_index]]
        X_sh = (brain[0] * self_payoff_matrix['CC']) + (brain[1] * self_payoff_matrix['CD'])
        X_st = (brain[0] * self_payoff_matrix['DC']) + (brain[1] * self_payoff_matrix['DD'])
        Action = reward - max(X_sh, X_st)
        kappa = 2.0
        A = 1 / (1 + mth.exp(-Action * kappa))
        if rnd.random() <= A:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Double Q-Learner':
        A_or_B = rnd.randint(0, 1)
        brain = internal_memory[180][A_or_B][0]
        another_brain = internal_memory[180][1 - A_or_B][0]
        another_factor = internal_memory[180][A_or_B][1]
        second_another_factor = internal_memory[180][1 - A_or_B][1]
        A = min(12, len(history)) - 1
        opp_inp = tuple(['C' for _ in range(12 - min(12, len(history)))] + [history[-i][opp_index] for i in range(min(12, len(history)), 0, -1)]) 
        self_inp = tuple(['C' for _ in range(12 - min(12, len(history)))] + [history[-i][self_index] for i in range(min(12, len(history)), 0, -1)]) 
        inp = (opp_inp, self_inp)
        if brain.get(inp) == None:
            brain[inp] = [0,0]
        if another_brain.get(inp) == None:
            another_brain[inp] = [0,0]
        if len(history) > 0 and another_factor[0][1] != 0.5:
            reward = 0
            if history[-1][opp_index] == 'C':
                if history[-1][self_index] == 'C':
                    reward = self_payoff_matrix['CC']
                else:
                    reward = self_payoff_matrix['DC']
            else:
                if history[-1][self_index] == 'C':
                    reward = self_payoff_matrix['CD']
                else:
                    reward = self_payoff_matrix['DD']
            c_factor = another_brain[inp][0]
            d_factor = another_brain[inp][1]
            a = 0.5
            gamma = 0.5
            Q_another = max(c_factor, d_factor)
            Q_old = brain[another_factor[0][0]][another_factor[0][1]]
            Q_new = Q_old + (a * (reward + ((gamma * Q_another) - Q_old)))
            brain[another_factor[0][0]][another_factor[0][1]] = Q_new
        if rnd.random() <= another_factor[1]:
            Action = rnd_prob(0.5)
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        decay = 1
        min_epsilon = 1
        another_factor[1] = max(another_factor[1] * decay, min_epsilon)
        if brain[inp][0] == brain[inp][1]:
            Action = rnd_prob(0.5)
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        best_choice = max(brain[inp])
        if best_choice == brain[inp][0]:
            another_factor[0] = [inp, 0]
            return 'C'
        else:
            another_factor[0] = [inp, 1]
            return 'D'
    elif strategy == 'Falsity':
        def clamp(x):
            return max(min(x, 1), 0)
        if internal_memory[181][3] > 0:
            internal_memory[181][3] -= 1
            return 'C'
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[181][0] -= 30
            else:
                internal_memory[181][0] += 5
        else:
            internal_memory[181][0] += 5
        if (internal_memory[181][0] < 30) or internal_memory[181][2]:
            if not internal_memory[181][2]:
                internal_memory[181][2] = True
            if internal_memory[181][1] < 2:
                internal_memory[181][1] = 2
            if (internal_memory[181][1] == 2) and (internal_memory[181][0] >= 30):
                internal_memory[181][2] = False
                internal_memory[181][3] = 1
                internal_memory[181][1] = 1
                return 'C'
        if internal_memory[181][1] == 1:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Cruelity':
        def clamp(x):
            return max(min(x, 1), 0)
        cruelity_eyes = [('C', 'C') for _ in range(10)] + history
        if internal_memory[182][4] > 0:
            internal_memory[182][4] -= 1
            return 'C'
        if cruelity_eyes[-1][opp_index] == 'D':
            internal_memory[182][0] -= 40
        else:
            internal_memory[182][0] += 5
        cruelity_cond = (cruelity_eyes[-1-1][opp_index], cruelity_eyes[-1-1][self_index])
        internal_memory[182][5][cruelity_cond][0] += (cruelity_eyes[-1][opp_index] == 'C')
        internal_memory[182][5][cruelity_cond][1] += 1
        if (internal_memory[182][0] < 30) or internal_memory[182][2]:
            if not internal_memory[182][2]:
                internal_memory[182][2] = True
                internal_memory[182][3] = (len(history) + 1)
            if internal_memory[182][1] < 2:
                internal_memory[182][1] = 2
            if (internal_memory[182][1] == 3) and (internal_memory[182][0] >= 30):
                internal_memory[182][2] = False
                internal_memory[182][4] = 1
                internal_memory[182][1] = 1
                return 'C'
            elif (((len(history) + 1) - internal_memory[182][3]) / 20) >= 0.75:
                internal_memory[182][1] = 3
        if internal_memory[182][1] == 1:
            return 'C'
        elif internal_memory[182][1] == 2:
            cyclic_inp = [cruelity_eyes[-i][opp_index] for i in range(1, 10 + 1)] 
            is_cycle = False
            if "D" in cyclic_inp:
                if cycle_detec(history[-10:], opp_index, min_ = 1, max_ = 5):
                    is_cycle = True
            if is_cycle:
                return 'D'
            mutual_cooperate_prob = ((internal_memory[182][5][("C", "C")][0] + 1) / (internal_memory[182][5][("C", "C")][1] + 2))
            exploitation_prob = ((internal_memory[182][5][("C", "D")][0] + 1) / (internal_memory[182][5][("C", "D")][1] + 2))
            exploited_prob = ((internal_memory[182][5][("D", "C")][0] + 1) / (internal_memory[182][5][("D", "C")][1] + 2))
            mutual_defect_prob = ((internal_memory[182][5][("D", "D")][0] + 1) / (internal_memory[182][5][("D", "D")][1] + 2))
            cruelity_category = [
                [ 
                    1,
                    0,
                    1,
                    0
                ],
                [ 
                    1,
                    0,
                    0,
                    1
                ],
                [ 
                    0,
                    0,
                    1,
                    1
                ]
            ]
            cruelity_enemy_differents = [0 for _ in range(len(cruelity_category))]
            for i in range(len(cruelity_category)):
                cruelity_enemy_differents[i] = (
                    abs(mutual_cooperate_prob - cruelity_category[i][0]) +
                    abs(exploitation_prob - cruelity_category[i][1]) +
                    abs(exploited_prob - cruelity_category[i][2]) +
                    abs(mutual_defect_prob - cruelity_category[i][3])
                ) / 4
            if min(cruelity_enemy_differents) <= 0.25:
                most_similar = [['Tit For Tat', 'Pavlov', 'Alternator'][j] for j in range(len(cruelity_enemy_differents)) if cruelity_enemy_differents[j] == min(cruelity_enemy_differents)]
                enemy_category = rnd.choice(most_similar)
                if enemy_category == 'Tit For Tat':
                    return 'C'
                elif enemy_category == 'Pavlov':
                    return ['D', 'D', 'C'][(len(history) + 1) % 3]
                else:
                    return 'D' if cruelity_eyes[-1][opp_index] == 'C' else rnd_prob(0.5)
            A = ((internal_memory[182][5][cruelity_cond][0] + 1) / (internal_memory[182][5][cruelity_cond][1] + 2))
            if rnd.random() <= A:
                return 'D' if rnd.random() <= ((internal_memory[182][5][("C", "D")][0] + 1) / (internal_memory[182][5][("C", "D")][1] + 2)) else 'C'
            else:
                return 'D'
        else:
            return 'D'
    elif strategy == 'Curiosity':
        def clamp(x):
            return max(min(x, 1), 0)
        curiosity_eyes = [('C', 'C') for _ in range(10)] + history
        if internal_memory[183][4] > 0:
            internal_memory[183][4] -= 1
            return 'C'
        if curiosity_eyes[-1][opp_index] == 'D':
            internal_memory[183][0] -= 30
        else:
            internal_memory[183][0] += 5
        curiosity_cond = (curiosity_eyes[-1-1][opp_index], curiosity_eyes[-1-1][self_index])
        internal_memory[183][5][curiosity_cond][0] += (curiosity_eyes[-1][opp_index] == 'C')
        internal_memory[183][5][curiosity_cond][1] += 1
        if (internal_memory[183][0] < 30) or internal_memory[183][2]:
            if not internal_memory[183][2]:
                internal_memory[183][2] = True
                internal_memory[183][3] = (len(history) + 1)
            if internal_memory[183][1] < 2:
                internal_memory[183][1] = 2
            if (internal_memory[183][1] == 3) and (internal_memory[183][0] >= 30):
                internal_memory[183][2] = False
                internal_memory[183][4] = 1
                internal_memory[183][1] = 1
                return 'C'
            elif (((len(history) + 1) - internal_memory[183][3]) / 20) >= 0.75:
                internal_memory[183][1] = 3
        if internal_memory[183][1] == 1:
            return 'C'
        elif internal_memory[183][1] == 2:
            mutual_cooperate_prob = ((internal_memory[183][5][("C", "C")][0] + 1) / (internal_memory[183][5][("C", "C")][1] + 2))
            exploitation_prob = ((internal_memory[183][5][("C", "D")][0] + 1) / (internal_memory[183][5][("C", "D")][1] + 2))
            exploited_prob = ((internal_memory[183][5][("D", "C")][0] + 1) / (internal_memory[183][5][("D", "C")][1] + 2))
            mutual_defect_prob = ((internal_memory[183][5][("D", "D")][0] + 1) / (internal_memory[183][5][("D", "D")][1] + 2))
            curiosity_category = [
                [ 
                    1,
                    0,
                    1,
                    0
                ],
                [ 
                    1,
                    0,
                    0,
                    1
                ],
                [ 
                    0,
                    0,
                    1,
                    1
                ],
                [ 
                    0,
                    1,
                    0,
                    1
                ],
                [ 
                    0.5,
                    0.5,
                    0.5,
                    0.5
                ]
            ]
            curiosity_enemy_differents = [0 for _ in range(len(curiosity_category))]
            for i in range(len(curiosity_category)):
                curiosity_enemy_differents[i] = (
                    abs(mutual_cooperate_prob - curiosity_category[i][0]) +
                    abs(exploitation_prob - curiosity_category[i][1]) +
                    abs(exploited_prob - curiosity_category[i][2]) +
                    abs(mutual_defect_prob - curiosity_category[i][3])
                ) / 4
            if min(curiosity_enemy_differents) <= 0.25:
                most_similar = [['Tit For Tat', 'Pavlov', 'Alternator', 'reverse tit for tat', 'Random'][j] for j in range(len(curiosity_enemy_differents)) if curiosity_enemy_differents[j] == min(curiosity_enemy_differents)]
                enemy_category = rnd.choice(most_similar)
                if enemy_category == 'Tit For Tat':
                    return 'C'
                elif enemy_category == 'Pavlov':
                    return ['D', 'D', 'C'][(len(history) + 1) % 3]
                elif enemy_category == 'Alternator':
                    return 'D' if curiosity_eyes[-1][opp_index] == 'C' else rnd_prob(0.5)
                elif enemy_category == 'reverse tit for tat':
                    return 'D'
                elif enemy_category == 'Random':
                    return 'D'
            A = clamp(((internal_memory[183][5][curiosity_cond][0] + 1) / (internal_memory[183][5][curiosity_cond][1] + 2)) - 0.25)
            if rnd.random() <= A:
                return 'C'
            else:
                return 'D'
        else:
            return 'D'
    elif strategy == 'Manipulity':
        def clamp(x):
            return max(min(x, 1), 0)
        manipulity_eyes = [('C', 'C') for _ in range(10)] + history
        if internal_memory[184][4] > 0:
            internal_memory[184][4] -= 1
            return 'C'
        if manipulity_eyes[-1][opp_index] == 'D':
            internal_memory[184][0] -= 30
        else:
            internal_memory[184][0] += 5
        manipulity_cond = (manipulity_eyes[-1-1][opp_index], manipulity_eyes[-1-1][self_index])
        internal_memory[184][5][manipulity_cond][0] += (manipulity_eyes[-1][opp_index] == 'C')
        internal_memory[184][5][manipulity_cond][1] += 1
        if (internal_memory[184][0] < 30) or internal_memory[184][2]:
            if not internal_memory[184][2]:
                internal_memory[184][2] = True
                internal_memory[184][3] = (len(history) + 1)
            if internal_memory[184][1] < 2:
                internal_memory[184][1] = 2
            if (internal_memory[184][1] == 3) and (internal_memory[184][0] >= 40):
                internal_memory[184][2] = False
                internal_memory[184][4] = 1
                internal_memory[184][1] = 1
                return 'C'
            elif (((len(history) + 1) - internal_memory[184][3]) / 20) >= 0.75:
                internal_memory[184][1] = 3
        if internal_memory[184][1] == 1:
            if rnd.random() <= clamp(((internal_memory[184][5][("C", "D")][0] + 1) / (internal_memory[184][5][("C", "D")][1] + 2)) - 0.25):
                return 'D'
            else:
                return 'C'
        elif internal_memory[184][1] == 2:
            cyclic_inp = [manipulity_eyes[-i][opp_index] for i in range(1, 10 + 1)] 
            is_cycle = False
            if "D" in cyclic_inp:
                if cycle_detec(history[-10:], opp_index, min_ = 1, max_ = 5):
                    is_cycle = True
            if is_cycle:
                return 'D'
            mutual_cooperate_prob = ((internal_memory[184][5][("C", "C")][0] + 1) / (internal_memory[184][5][("C", "C")][1] + 2))
            exploitation_prob = ((internal_memory[184][5][("C", "D")][0] + 1) / (internal_memory[184][5][("C", "D")][1] + 2))
            exploited_prob = ((internal_memory[184][5][("D", "C")][0] + 1) / (internal_memory[184][5][("D", "C")][1] + 2))
            mutual_defect_prob = ((internal_memory[184][5][("D", "D")][0] + 1) / (internal_memory[184][5][("D", "D")][1] + 2))
            manipulity_category = [
                [ 
                    1,
                    0,
                    1,
                    0
                ],
                [ 
                    1,
                    0,
                    0,
                    1
                ],
                [ 
                    0,
                    0,
                    1,
                    1
                ],
                [ 
                    0,
                    1,
                    0,
                    1
                ],
                [ 
                    0.5,
                    0.5,
                    0.5,
                    0.5
                ]
            ]
            manipulity_enemy_differents = [0 for _ in range(len(manipulity_category))]
            for i in range(len(manipulity_category)):
                manipulity_enemy_differents[i] = (
                    abs(mutual_cooperate_prob - manipulity_category[i][0]) +
                    abs(exploitation_prob - manipulity_category[i][1]) +
                    abs(exploited_prob - manipulity_category[i][2]) +
                    abs(mutual_defect_prob - manipulity_category[i][3])
                ) / 4
            if min(manipulity_enemy_differents) <= 0.25:
                most_similar = [['Tit For Tat', 'Pavlov', 'Alternator', 'reverse tit for tat', 'Random'][j] for j in range(len(manipulity_enemy_differents)) if manipulity_enemy_differents[j] == min(manipulity_enemy_differents)]
                enemy_category = rnd.choice(most_similar)
                if enemy_category == 'Tit For Tat':
                    return 'C'
                elif enemy_category == 'Pavlov':
                    return ['D', 'D', 'C'][(len(history) + 1) % 3]
                elif enemy_category == 'Alternator':
                    return 'D' if manipulity_eyes[-1][opp_index] == 'C' else rnd_prob(0.5)
                elif enemy_category == 'reverse tit for tat':
                    return 'D'
                elif enemy_category == 'Random':
                    return 'D'
            A = ((internal_memory[184][5][manipulity_cond][0] + 1) / (internal_memory[184][5][manipulity_cond][1] + 2))
            if rnd.random() <= A:
                if rnd.random() <= clamp(((internal_memory[184][5][("C", "D")][0] + 1) / (internal_memory[184][5][("C", "D")][1] + 2)) - 0.25):
                    return 'D'
                else:
                    return 'C'
            else:
                return 'D'
        else:
            return 'D'
    elif strategy == 'Charity':
        def clamp(x):
            return max(min(x, 1), 0)
        if internal_memory[185][3] > 0:
            internal_memory[185][3] -= 1
            return 'C'
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[185][0] -= 30
            else:
                internal_memory[185][0] += 5
                internal_memory[185][4] += 1
        else:
            internal_memory[185][0] += 5
            internal_memory[185][4] += 1
        if (internal_memory[185][0] < 30) or internal_memory[185][2]:
            if not internal_memory[185][2]:
                internal_memory[185][2] = True
            if internal_memory[185][1] < 2:
                internal_memory[185][1] = 2
            if (internal_memory[185][1] == 2) and (internal_memory[185][0] >= 30):
                internal_memory[185][2] = False
                internal_memory[185][3] = 1
                internal_memory[185][1] = 1
                return 'C'
        if internal_memory[185][1] == 1:
            return 'C'
        else:
            return 'C' if rnd.random() <= (internal_memory[185][4] / (len(history) + 1)) else 'D'
    elif strategy == 'Calamity':
        def clamp(x):
            return max(min(x, 1), 0)
        calamity_eyes = [('C', 'C') for _ in range(10)] + history
        if internal_memory[186][4] > 0:
            internal_memory[186][4] -= 1
            return 'C'
        if calamity_eyes[-1][opp_index] == 'D':
            internal_memory[186][0] -= 30
        else:
            internal_memory[186][0] += 5
        calamity_cond = (calamity_eyes[-1-1][opp_index], calamity_eyes[-1-1][self_index])
        internal_memory[186][5][calamity_cond][0] += (calamity_eyes[-1][opp_index] == 'C')
        internal_memory[186][5][calamity_cond][1] += 1
        if (internal_memory[186][0] < 30) or internal_memory[186][2]:
            if not internal_memory[186][2]:
                internal_memory[186][2] = True
                internal_memory[186][3] = (len(history) + 1)
            if internal_memory[186][1] < 2:
                internal_memory[186][1] = 2
            if (internal_memory[186][1] == 3) and (internal_memory[186][0] >= 30):
                internal_memory[186][2] = False
                internal_memory[186][4] = 1
                internal_memory[186][1] = 1
                return 'C'
            elif (((len(history) + 1) - internal_memory[186][3]) / 20) >= 0.75:
                internal_memory[186][1] = 3
        if internal_memory[186][1] == 1:
            if rnd.random() <= clamp(((internal_memory[186][5][("C", "D")][0] + 1) / (internal_memory[186][5][("C", "D")][1] + 2)) - 0.125):
                return 'D'
            else:
                return 'C'
        elif internal_memory[186][1] == 2:
            mutual_cooperate_prob = ((internal_memory[186][5][("C", "C")][0] + 1) / (internal_memory[186][5][("C", "C")][1] + 2))
            exploitation_prob = ((internal_memory[186][5][("C", "D")][0] + 1) / (internal_memory[186][5][("C", "D")][1] + 2))
            exploited_prob = ((internal_memory[186][5][("D", "C")][0] + 1) / (internal_memory[186][5][("D", "C")][1] + 2))
            mutual_defect_prob = ((internal_memory[186][5][("D", "D")][0] + 1) / (internal_memory[186][5][("D", "D")][1] + 2))
            calamity_category = [
                [ 
                    1,
                    0,
                    1,
                    0
                ],
                [ 
                    1,
                    0,
                    0,
                    1
                ],
                [ 
                    0,
                    0,
                    1,
                    1
                ],
                [ 
                    0,
                    1,
                    0,
                    1
                ]
            ]
            calamity_enemy_differents = [0 for _ in range(len(calamity_category))]
            for i in range(len(calamity_category)):
                calamity_enemy_differents[i] = (
                    abs(mutual_cooperate_prob - calamity_category[i][0]) +
                    abs(exploitation_prob - calamity_category[i][1]) +
                    abs(exploited_prob - calamity_category[i][2]) +
                    abs(mutual_defect_prob - calamity_category[i][3])
                ) / 4
            if min(calamity_enemy_differents) <= 0.25:
                most_similar = [['Tit For Tat', 'Pavlov', 'Alternator', 'reverse tit for tat'][j] for j in range(len(calamity_enemy_differents)) if calamity_enemy_differents[j] == min(calamity_enemy_differents)]
                enemy_category = rnd.choice(most_similar)
                if enemy_category == 'Tit For Tat':
                    return 'C'
                elif enemy_category == 'Pavlov':
                    return ['D', 'D', 'C'][(len(history) + 1) % 3]
                elif enemy_category == 'Alternator':
                    return 'D' if calamity_eyes[-1][opp_index] == 'C' else rnd_prob(0.5)
                elif enemy_category == 'reverse tit for tat':
                    return 'D'
            A = ((internal_memory[186][5][calamity_cond][0] + 1) / (internal_memory[186][5][calamity_cond][1] + 2))
            if rnd.random() <= A:
                return 'D' if rnd.random() <= clamp(((internal_memory[186][5][("C", "D")][0] + 1) / (internal_memory[186][5][("C", "D")][1] + 2)) - 0.125) else 'C'
            else:
                return 'D'
        else:
            return 'D'
    elif strategy == 'Amazon / Piraha':
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[187] += 1
        else:
            return 'C'
        B = min(2, len(history))
        inp = tuple([history[-i][opp_index] for i in range(B, 0, -1)])
        if inp == tuple(['C'] * B):
            internal_memory[187] -= 1
        internal_memory[187] = max(internal_memory[187], 0)
        maximal_few = rnd.randint(2, 3)
        maximal_many = rnd.randint(5, 6)
        if internal_memory[187] <= maximal_few: 
            return 'C'
        elif internal_memory[187] <= maximal_many: 
            return rnd_prob(0.5)
        else: 
            return 'D'
    elif strategy == 'Mosquito':
        def clamp(x):
            return max(min(x, 1), 0)
        cond = (history[-1-1][opp_index], history[-1-1][self_index]) if (len(history) + 1) > 2 else ('C', 'C')
        internal_memory[188][cond][0] += (history[-1][opp_index] == 'C') if len(history) > 0 else True
        internal_memory[188][cond][1] += 1
        A = clamp(((internal_memory[188][cond][0] + 1) / (internal_memory[188][cond][1] + 2)) - 0.25)
        if rnd.random() <= A:
            if rnd.random() <= clamp(((internal_memory[188][("C", "D")][0] + 1) / (internal_memory[188][("C", "D")][1] + 2)) - 0.25):
                return 'D'
            else:
                return 'C'
        else:
            return 'D'
    elif strategy == 'Echo Fisher':
        if not len(history):
            return 'C'
        if (len(history) + 1) % 2 == 0:
            return 'D' if history[-1][opp_index] == 'C' else "C"
        else:
            return history[-1][opp_index]
    elif strategy == 'Devil Staircase':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'D':
            internal_memory[189] += 1
        def devil_staircase(x, iterations=5):
            """
            Menghitung nilai Devil's Staircase untuk SATU angka tunggal (skalar).
            x: angka antara 0 dan 1 (contoh: 0.25, 0.75).
            iterations: jumlah detail iterasi fraktal.
            """
            if x <= 0.0: return 0.0
            if x >= 1.0: return 1.0
            y = 0.0
            x_current = float(x)
            for i in range(1, iterations + 1):
                digit = np.floor(x_current * 3)
                if digit == 1:
                    y += 1.0 / (2**i)
                    break 
                elif digit >= 2:
                    y += 1.0 / (2**i)
                    x_current = 3 * x_current - 2
                else:
                    x_current = 3 * x_current
            return y
        p_d = internal_memory[189] / (len(history))
        if rnd.random() <= devil_staircase(p_d):
            return 'D'
        else:
            return 'C'
    elif strategy == 'Hot Coffee':
        def clamp(x):
            return max(min(x, 1), 0)
        if len(history) > 0:
            if history[-1][opp_index] == 'C':
                internal_memory[190] += 1
            else:
                internal_memory[190] -= 3
        else:
            internal_memory[190] += 1
        internal_memory[190] = max(min(internal_memory[190], 100), 0)
        energy = clamp(internal_memory[190] / 100)
        caffeine = 1 - (((len(history) + 1) % 10) / 10)
        if rnd.random() <= (energy * (1 + caffeine)):
            if not len(history):
                return 'C'
            return history[-1][opp_index]
        else:
            return 'C'
    elif strategy == 'The Cave Breaker':
        the_cave_breaker_eyes = [('C', 'C') for _ in range(3)] + history
        if the_cave_breaker_eyes[-1][opp_index] == 'D':
            internal_memory[196][0] += 1
        if the_cave_breaker_eyes[-1][self_index] == 'D':
            internal_memory[196][2] += 1
        if (len(history) + 1) > 11 and (internal_memory[196][0] / (len(history))) > 0.6:
            return 'D'
        inp = tuple([the_cave_breaker_eyes[-i][opp_index] for i in range(3, 0, -1)]) 
        if internal_memory[196][1] or ((len(history) + 1) > 6 and inp == ('D', 'D', 'D')):
            internal_memory[196][1] = True
            return 'D'
        if (len(history)) % 5 == 0 and internal_memory[196][2] < 15:
            return 'D'
        if len(history) >= (tournament_avg_last_round - 15):
            return 'D'
        return the_cave_breaker_eyes[-1][opp_index]
    elif strategy == 'Random\'s Exploiter 2':
        if not len(history): return 'C'
        if history[-1][opp_index] == 'D': internal_memory[198] += 1
        def sin_squared(x): return mth.sin(mth.pi * x) * mth.sin(mth.pi * x)
        p_d = internal_memory[198] / (len(history))
        if abs(sin_squared(p_d) - 1) <= 0.1: return 'D'
        return history[-1][opp_index]
    elif strategy == 'George':
        def clamp(x): return max(min(x, 1), 0)
        A = tournament_avg_last_round / 2
        return 'D' if rnd.random() <= (0.2 * (((len(history) + 1) - A) / A)) else (history[-1][opp_index] if len(history) > 0 else 'C')
    elif strategy == 'Long Horse':
        def clamp(x): return max(min(x, 1), 0)
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[199][0] += 0.1
                internal_memory[199][1] = 0
                if (score[self_index] - score[opp_index]) >= 0:
                    return 'C'
                else:
                    return 'D'
        if internal_memory[199][1] < 4:
            if (len(history) + 1) % 50 == 0:
                internal_memory[199][1] += 1
                return 'D'
            if rnd.random() <= clamp(internal_memory[199][0]):
                return 'D'
        if not len(history):
            return 'C'
        return history[-1][opp_index]
    elif strategy == 'The Nasty A':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'D':
            internal_memory[201] = 1
        if (internal_memory[201] == 1) or (len(history) >= (tournament_avg_last_round)):
            return 'D'
        return 'C'
    elif strategy == 'Monkey See, Monkey Do':
        if (len(history)) < 3:
            if not len(history):
                return 'D'
            else:
                return history[-1][opp_index]
        inp = tuple([history[-i][opp_index] for i in range(3, 0, -1)]) 
        if inp == ('C', 'C', 'C'):
            internal_memory[202] = 1
        if internal_memory[202] == 1:
            return 'C'
        else:
            return history[-1][opp_index]
    elif strategy == 'Genetic Algo 1':
        table = {
            'CC':'C',
            'CD':'D',
            'DC':'C',
            'DD':'C'
        }
        inp = (history[-1][self_index] + history[-1][opp_index]) if len(history) > 0 else 'CC'
        return table[inp]
    elif strategy == 'Genetic Algo 2':
        table = {
            'CCCC':'C',
            'CCCD':'D',
            'CCDC':'C',
            'CCDD':'C',
            'CDCC':'C',
            'CDCD':'D',
            'CDDC':'C',
            'CDDD':'C',
            'DCCC':'C',
            'DCCD':'D',
            'DCDC':'C',
            'DCDD':'C',
            'DDCC':'C',
            'DDCD':'D',
            'DDDC':'C',
            'DDDD':'D'
        }
        inp = ((history[-1 - 1][self_index] + history[-1 - 1][opp_index]) if (len(history) + 1) > 2 else 'CC') + ((history[-1][self_index] + history[-1][opp_index]) if len(history) > 0 else 'CC')
        return table[inp]
    elif strategy == 'Random\'s Exploiter 3':
        if not len(history): return 'C'
        if history[-1][opp_index] == 'D': internal_memory[215] += 1
        p_d = internal_memory[215] / (len(history))
        if abs(p_d - 0.5) <= 0.1: return 'D'
        return history[-1][opp_index]
    elif strategy == 'Gambler Attariq Version':
        if score[self_index] >= 15:
            if rnd.random() <= ((len(history) + 1) / tournament_avg_last_round):
                lose_gambling = (rnd.random() > 0.1)
                if lose_gambling and (internal_memory[37] == 0):
                    internal_memory[37] = 3
                    return 'C'
        if internal_memory[37] > 0:
            internal_memory[37] -= 1
            return 'D'
        else:
            if not len(history):
                return 'C'
            return history[-1][opp_index]
    elif strategy == 'Random\'s Exploiter 4':
        stev = 0.4
        mean = 2.0
        if not len(history):
            internal_memory[102][1] = 1 / (stev * ((2 * mth.pi) ** 0.5))
            internal_memory[102][2] = mean / stev
            return 'C'
        if history[-1][opp_index] == 'D':
            internal_memory[102][0] += 1
        p_d = internal_memory[102][0] / (len(history))
        A1 = ((p_d - 0.5) * internal_memory[102][2])
        A2 = internal_memory[102][1] * mth.exp(-((A1 * A1) / 2))
        if abs(A2 - 1) <= 0.2: return 'D'
        return history[-1][opp_index]
    elif strategy == 'Toxic Mirror':
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[76] += 1
        if (len(history)) < 4:
            return 'C'
        p_d = internal_memory[76] / (len(history))
        if p_d > 0.25:
            return 'C'
        if (len(history)) % 7 == 0:
            return 'D'
        else:
            if history[-1][opp_index] == 'D':
                if rnd.random() < 0.3:
                    return 'C'
                else:
                    return 'D'
            else:
                return 'C'
    elif strategy == 'Sweet Heart':
        if (len(history)) < 4:
            return 'C'
        if (len(history) + 1) == 11:
            return 'D'
        if (len(history) + 1) > (tournament_avg_last_round - 20):
            if (len(history) + 1) % 3 == 0:
                return 'D'
        if history[-1][opp_index] == 'D':
            if rnd.random() < 0.6:
                return 'C'
            else:
                return 'D'
        else:
            return 'C'
    elif strategy == 'Split Brain Syndrome':
        if not len(history):
            internal_memory[103][0] = 0   
            internal_memory[103][1] = 0   
            internal_memory[103][2] = 0   
            internal_memory[103][3] = 0   
        lawan = history[-1][opp_index] if len(history) > 0 else 'C'
        kiri = 'C'
        kanan = 'C'
        if internal_memory[103][0] > 2:
            kiri = 'D'
        if internal_memory[103][1] >= 3:
            kanan = 'D'
        if lawan == 'D':
            internal_memory[103][0] += 1
            internal_memory[103][1] += 1
        else:
            internal_memory[103][1] = max(0, internal_memory[103][1] - 1)
        if kiri == kanan:
            pilihan = kiri
        else:
            pilihan = 'C'  
        return pilihan
    elif strategy == 'Golden Mean':
        if not len(history):
            internal_memory[216] = 0  
        if len(history) > 0:
            if history[-1][opp_index] == 'C':
                internal_memory[216] += 1
        else:
            internal_memory[216] += 1
        c_count = internal_memory[216]
        total = len(history)
        rasio = c_count / total if total > 0 else 1.0
        if rasio >= 0.80:
            return 'C'  
        else:
            return history[-1][opp_index]
    elif strategy == 'Patterned Adaptive Fortress 2':
        state = internal_memory[217]
        alpha_base = 0.08
        lawan = history[-1][opp_index] if len(history) > 0 else 'C'
        if (len(history)) < 3:
            state["last_my_action"] = 'C'
            return 'C'
        b = 1.0 if lawan == 'C' else 0.0
        c = state["ema_trust"]
        alpha = alpha_base
        if state["last_my_action"] == 'C' and state["d_streak"] > 0 and lawan == 'D':
            state["regret_count"] += 1
            alpha = 0.25
        state["ema_trust"] = (alpha * b) + ((1 - alpha) * c)
        if lawan == 'C' and state["d_streak"] == 0:
            state["regret_count"] = max(0, state["regret_count"] - 0.005)
            state["pattern_score"] = max(0, state["pattern_score"] - 0.002)
        state["history"].append(b)
        if len(state["history"]) > 8:
            state["history"].pop(0)
        recent_trust = sum(state["history"]) / len(state["history"])
        if len(state["history"]) >= 4:
            history_str = ''.join(['C' if x == 1 else 'D' for x in state["history"]])
            if history_str[-4:] in ["CDCD", "DCDC", "DDCD"]:
                if state["pattern_window"] == 0:
                    state["pattern_score"] = min(2.0, state["pattern_score"] + 0.5)
                    state["pattern_window"] = 4
            if len(state["history"]) >= 6 and history_str[-6:] in ["CCDCCD", "DCCDCC"]:
                if state["pattern_window"] == 0:
                    state["pattern_score"] = min(2.0, state["pattern_score"] + 0.5)
                    state["pattern_window"] = 6
        if state["pattern_window"] > 0:
            state["pattern_window"] -= 1
        volatility = abs(state["ema_trust"] - recent_trust)
        a = max(0.3, min(0.9, 1 - volatility))
        combined_trust = (a * state["ema_trust"]) + ((1 - a) * recent_trust)
        state["d_streak"] = state["d_streak"] + 1 if lawan == 'D' else 0
        if (len(history) + 1) >= (tournament_avg_last_round - 5):
            if combined_trust < 0.85 or lawan == 'D':
                state["last_my_action"] = 'D'
                return 'D'
        if state["d_streak"] >= 3:
            state["last_my_action"] = 'D'
            return 'D'
        if state["punish_cooldown"] > 0:
            state["punish_cooldown"] -= 1
            state["last_my_action"] = 'D'
            return 'D'
        if lawan == 'D' and state["d_streak"] >= 2 and combined_trust < 0.7:
            state["punish_cooldown"] = 2
            state["last_my_action"] = 'D'
            return 'D'
        total_suspicion = state["regret_count"] + state["pattern_score"]
        required_trust = min(0.90, 0.60 + (total_suspicion * 0.10))
        if combined_trust < required_trust:
            my_action = 'D' if lawan == 'D' else 'C'
        else:
            my_action = 'C'
        state["last_my_action"] = my_action
        return my_action
    elif strategy == 'Perceptron':
        brain = internal_memory[218]
        if len(history) > 0:
            R_or_P = 0
            if history[-1][opp_index] == 'C':
                if history[-1][self_index] == 'C':
                    R_or_P = self_payoff_matrix['CC']
                else:
                    R_or_P = self_payoff_matrix['DC']
            else:
                if history[-1][self_index] == 'C':
                    R_or_P = -4 * (self_payoff_matrix['DD'] - self_payoff_matrix['CD'])
                else:
                    R_or_P = -max(abs(self_payoff_matrix['DD']), 1)
            learning_rate = 0.5
            A = R_or_P * learning_rate
            for i in range(brain["n"]):
                brain["input weight"][i] = mth_mtx.matrix_aritc(brain["input weight"][i], mth_mtx.matrix_function(brain["old input"], A, '*'), '+')
            for i in range(2):
                brain["output weight"][i] = mth_mtx.matrix_aritc(brain["output weight"][i], mth_mtx.matrix_function(brain["n val"], A, '*'), '+')
        inp = [(history[-1][opp_index] == 'C'), (history[-1][self_index] == 'C')] if len(history) > 0 else [True, True]
        brain["old input"] = [(history[-1][opp_index] == 'C'), (history[-1][self_index] == 'C')] if len(history) > 0 else [True, True]
        brain["n val"] = mth_mtx.matrix_function(mth_mtx.matrix_aritc(mth_mtx.matrix_list_nestlist(inp, brain["input weight"], brain["n"], None, 'dot prod'), brain["bias"], '+'), (0, 1, 0), 'step')
        brain["output val"] = mth_mtx.matrix_aritc(mth_mtx.matrix_list_nestlist(brain["n val"], brain["output weight"], 2, None, 'dot prod'), brain["output bias"], '+')
        if brain["output val"][0] >= brain["output val"][1]:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Simple ANN':
        brain = internal_memory[219]
        max_time = 50
        if brain["time"] > max_time:
            for i in range(brain["n"]):
                brain["input weight"][i] = [j + rnd.uniform(-0.15, 0.15) for j in brain["best input weight"][i]]
            for i in range(2):
                brain["output weight"][i] = [j + rnd.uniform(-0.15, 0.15) for j in brain["best output weight"][i]]
            brain["bias"] = [j + rnd.uniform(-0.15, 0.15) for j in brain["best bias"]]
            brain["output bias"] = [j + rnd.uniform(-0.15, 0.15) for j in brain["best output bias"]]
            brain["time"] = 0
        if len(history) > 0:
            learning_rate = 0.1
            if brain["reward"] > 0:
                A = 1 * learning_rate
            else:
                A = -1 * learning_rate
            brain["input weight"] = mth_mtx.matrix_function(brain["input weight"], A, '+')
            brain["output weight"] = mth_mtx.matrix_function(brain["output weight"], A, '+')
            brain["bias"] = mth_mtx.matrix_function(brain["bias"], A, '+')
            brain["output bias"] = mth_mtx.matrix_function(brain["output bias"], A, '+')
            if history[-1][opp_index] == 'C':
                brain["reward"] += 1
                brain["score"] += 1
            else:
                brain["reward"] -= 1
                brain["score"] -= 1
        inp = [(history[-1][opp_index] == 'C'), (history[-1][self_index] == 'C')] if len(history) > 0 else [True, True]
        if brain["score"] > brain["high score"]:
            brain["high score"] = brain["score"]
            for i in range(brain["n"]):
                brain["best input weight"][i] = brain["input weight"][i][:]
            for i in range(2):
                brain["best output weight"][i] = brain["output weight"][i][:]
            brain["best bias"] = brain["bias"][:]
            brain["best output bias"] = brain["output bias"][:]
        brain["n val"] = mth_mtx.matrix_function(mth_mtx.matrix_aritc(mth_mtx.matrix_list_nestlist(inp, brain["input weight"], brain["n"], None, 'dot prod'), brain["bias"], '+'), (0, 1, 0), 'step')
        brain["output val"] = mth_mtx.matrix_aritc(mth_mtx.matrix_list_nestlist(brain["n val"], brain["output weight"], 2, None, 'dot prod'), brain["output bias"], '+')
        brain["time"] += 1
        if brain["output val"][0] >= brain["output val"][1]:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Whale':
        state = internal_memory[220]
        lawan = history[-1][opp_index] if len(history) > 0 else 'C'
        if (len(history)) < 7:
            if (len(history)) < 5:
                state["history"].append(lawan)
                if state["history"].count('D') >= 3:
                    return 'D'
            return 'C'
        state["history"].append(lawan)
        if len(state["history"]) > 10:
            state["history"].pop(0)
        history_str = ''.join(state["history"])
        if history_str[-4:] in ["CDCD", "DCDC"]:
            state["pattern_score"] += 0.3
        if lawan == 'D':
            state["d_streak"] += 1
            state["trust"] = max(0, state["trust"] - 0.1)
        else:
            state["d_streak"] = 0
            state["trust"] = min(1, state["trust"] + 0.05)
        if state["d_streak"] >= 3 or state["trust"] < 0.4:
            return 'D'
        if state["pattern_score"] > 1.5:
            return 'D'
        if lawan == 'D' and rnd.random() < state["forgiveness_chance"]:
            return 'C'
        return lawan  
    elif strategy == 'Rice Field':
        if (len(history) + 1) < 12:
            return 'C'
        if internal_memory[221][0] == 1:
            if ((len(history) + 1) - internal_memory[221][1] > 25) and \
               (history[-1][opp_index] == 'C' and history[-2][opp_index] == 'C' and history[-3][opp_index] == 'C'):
                internal_memory[221][0] = 0
                internal_memory[221][1] = 0
                return 'C'
            return history[-1][opp_index]
        if (len(history) + 1) % 12 == 0:
            internal_memory[221][2] += 1 
            return 'D'
        if (len(history) + 1) % 12 == 1:
            if history[-1][self_index] == 'D' and history[-1][opp_index] == 'D':
                internal_memory[221][0] = 1
                internal_memory[221][1] = (len(history) + 1)
        return 'C'
    elif strategy == '3 Seconds Mirror':
        if (len(history) + 1) < 4:
            return 'C'
        delayed_move = history[-3][opp_index]
        if history[-1][opp_index] == 'C' and history[-2][opp_index] == 'C' and delayed_move == 'D':
            return 'C' 
        return delayed_move
    elif strategy == 'Caveman':
        if len(history) > 0:
            internal_memory[222][1] *= 0.9
            internal_memory[222][0] *= 0.9
            if history[-1][opp_index] == 'D':
                internal_memory[222][0] += 1
                internal_memory[222][1] += 0.1
                if internal_memory[222][2] == 0:
                    internal_memory[222][2] = round(internal_memory[222][0])
                else:
                    internal_memory[222][2] += 1
        next_chat[0] = 'UGA BUGA!'
        if rnd.random() <= max(internal_memory[222][1] - 0.2, 0):
            return 'D'
        if rnd.random() <= max(0.2 - internal_memory[222][1], 0):
            return 'D'
        if internal_memory[222][2] > 0:
            internal_memory[222][2] -= 1
            return 'D'
        else:
            return 'C'
    elif strategy == 'Slanger':
        slanger_eyes = [('C', 'C') for _ in range(2)] + history
        internal_memory[223] *= 0.9
        if slanger_eyes[-1][opp_index] == 'C' and slanger_eyes[-1][self_index] == 'D':
            if slanger_eyes[-2][opp_index] == 'D' and slanger_eyes[-2][self_index] == 'C':
                next_chat[0] = 'AHA! WHO\'S LAUGHING NOW BUDDY? WHO\'S LAUGHING NOW!!! >:-)'
                internal_memory[223] += 0.1
            elif slanger_eyes[-2][opp_index] == 'D' and slanger_eyes[-2][self_index] == 'D':
                next_chat[0] = 'YOU, FOOL!!! >:-)'
                internal_memory[223] += 0.15
            elif slanger_eyes[-2][opp_index] == 'C' and slanger_eyes[-2][self_index] == 'D':
                next_chat[0] = 'AHA! You are cooked again >:-)'
                internal_memory[223] += 0.175
            else:
                next_chat[0] = 'AHA! You are cooked >:-)'
                internal_memory[223] += 0.2
        if slanger_eyes[-1][opp_index] == 'D' and slanger_eyes[-1][self_index] == 'C':
            if slanger_eyes[-2][opp_index] == 'D' and slanger_eyes[-2][self_index] == 'C':
                next_chat[0] = 'I SO HATE YOU!!! >:-('
                internal_memory[223] += 0.25
            elif slanger_eyes[-2][opp_index] == 'D' and slanger_eyes[-2][self_index] == 'D':
                next_chat[0] = 'OH, CRAP! >:-('
                internal_memory[223] += 0.175
            elif slanger_eyes[-2][opp_index] == 'C' and slanger_eyes[-2][self_index] == 'D':
                next_chat[0] = 'I\'m sorry! :-('
                internal_memory[223] += -0.1
            else:
                next_chat[0] = 'I HATE YOU! >:-('
                internal_memory[223] += 0.2
        if slanger_eyes[-1][opp_index] == 'C' and slanger_eyes[-1][self_index] == 'C':
            if slanger_eyes[-2][opp_index] == 'D' and slanger_eyes[-2][self_index] == 'C':
                next_chat[0] = 'I don\'t too trust you, but okay :-('
                internal_memory[223] += 0.05
            elif slanger_eyes[-2][opp_index] == 'D' and slanger_eyes[-2][self_index] == 'D':
                next_chat[0] = 'I don\'t too trust you, but okay :-|'
                internal_memory[223] += 0.0
            elif slanger_eyes[-2][opp_index] == 'C' and slanger_eyes[-2][self_index] == 'D':
                next_chat[0] = 'YES! We can become friends again! >:-)'
                internal_memory[223] += 0.1
            else:
                next_chat[0] = 'Were friends! :-)'
                internal_memory[223] += -0.1
        if slanger_eyes[-1][opp_index] == 'D' and slanger_eyes[-1][self_index] == 'D':
            if slanger_eyes[-2][opp_index] == 'D' and slanger_eyes[-2][self_index] == 'C':
                next_chat[0] = 'TRAITOR!!!!! >:-('
                internal_memory[223] += 0.25
            elif slanger_eyes[-2][opp_index] == 'D' and slanger_eyes[-2][self_index] == 'D':
                next_chat[0] = 'NOT AGAIN! >:-('
                internal_memory[223] += 0.2
            elif slanger_eyes[-2][opp_index] == 'C' and slanger_eyes[-2][self_index] == 'D':
                next_chat[0] = 'OKAY! IF YOU WAN\'T TO FIGHT ME THEN! >:-)'
                internal_memory[223] += 0.15
            else:
                next_chat[0] = 'OH COME ON! >:-('
                internal_memory[223] += 0.25
        internal_memory[223] = max(min(internal_memory[223], 1), 0)
        if rnd.random() <= internal_memory[223]:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Chatter':
        lawan = history[-1][opp_index] if len(history) > 0 else 'C'
        aku = history[-1][self_index] if len(history) > 0 else 'C'
        pesan_lawan = chats[opp_index]  
        if 'cooked' in pesan_lawan or 'AHA' in pesan_lawan:
            next_chat[0] = "Don't be so arrogant! 😤 You will lose soon! 😏💪"
        elif lawan == 'D' and aku == 'C':
            next_chat[0] = "Why did you betray me?! 😢💔 Please don't do that again! 🙏"
        elif lawan == 'C':
            next_chat[0] = "Good! 🤝 Let's keep it this way! 😊✨"
        else:
            next_chat[0] = "Enough?! 😤 Let's make peace! 🤍🕊️"
        if lawan == 'D':
            return 'D'
        return 'C'
    elif strategy == 'Self-Love A-Learner':
        exploitation_rate = 1
        exploration_rate = 0.1
        if len(history) > 0:
            old_cond = internal_memory[91][1]
            reward = (score[self_index] - internal_memory[91][2])
            internal_memory[91][0][old_cond][(history[-1][self_index] == 'D')] += reward * exploitation_rate
        cond = (history[-1][self_index] + history[-1][opp_index]) if len(history) > 0 else 'CC'
        internal_memory[91][1] = cond
        internal_memory[91][2] = score[self_index]
        if (rnd.random() <= exploration_rate) or (internal_memory[91][0][cond][0] == internal_memory[91][0][cond][1]):
            return rnd.choice(history[-1]) if len(history) > 0 else 'C'
        if internal_memory[91][0][cond][0] > internal_memory[91][0][cond][1]:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Diplomat':
        state = internal_memory[224]
        if not len(history):
            next_chat[0] = "Let's cooperate. If you betray me, I'll punish you."
            state["coop_count"] += 1
            state["trust"] = min(1.0, state["trust"] + 0.05)
        else:
            if history[-1][opp_index] == 'C':
                state["coop_count"] += 1
                state["trust"] = min(1.0, state["trust"] + 0.05)
            else:
                state["defect_count"] += 1
                state["trust"] = max(0.0, state["trust"] - 0.15)
                if state["trust"] < 0.4:
                    state["deal_broken"] = True
        if state["deal_broken"]:
            if (len(history) + 1) == int(tournament_avg_last_round * 0.7):
                next_chat[0] = "You broke the deal."
            return 'D'
        if state["trust"] >= 0.6:
            if (len(history) + 1) % 20 == 0:
                next_chat[0] = "Good, let's keep this up."
            return 'C'
        else:
            if not len(history):
                return 'C'
            return history[-1][opp_index]
    elif strategy == 'Bluff Master':
        state = internal_memory[225]
        if (len(history) + 1) % 15 == 1:
            next_chat[0] = rnd.choice([
                "I will always cooperate, trust me.",
                "No hard feelings, just business.",
                "We can be friends, I promise."
            ])
        opp_chat = chats[opp_index].lower()
        if "cooperate" in opp_chat or "friend" in opp_chat or "trust" in opp_chat:
            state["opp_coop_words"] += 1
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                if state["opp_coop_words"] > 0:
                    state["opp_betrayals"] += 1
        opp_bluff_ratio = (
            state["opp_betrayals"] / max(1, state["opp_coop_words"])
        )
        base_move = 'C'
        bluff_chance = 0.05 + 0.1 * min(1.0, opp_bluff_ratio)
        if rnd.random() < bluff_chance:
            state["my_bluff_count"] += 1
            return 'D'
        if (len(history) + 1) > int(tournament_avg_last_round * 0.7):
            if opp_bluff_ratio > 0.5:
                return 'D'
        return base_move
    elif strategy == 'Is There Something New?':
        def clamp(x):
            return max(min(x, 1), 0)
        if len(history) > 0:
            if history[-1][opp_index] == 'C':
                internal_memory[226][0] += 1
        if (len(history)) < 5:
            return 'C'
        A1 = internal_memory[226][0] / (len(history))
        A2 = sum((i[opp_index] == 'C') for i in history[-5:]) / 5
        if abs(A1 - A2) >= 0.1:
            internal_memory[226][1].pop(0)
            internal_memory[226][1].append(A2 - A1)
        A = 1 - clamp(mth_mtx.matrix_dot_prod(internal_memory[226][1], [1 / (2 ** (i + 1)) for i in range(5)]))
        if rnd.random() <= A:
            return history[-1][opp_index]
        return history[-1][self_index]
    elif strategy == 'Forest-XG':
        brain = internal_memory[227]
        cond = (history[-1][self_index] + history[-1][opp_index]) if len(history) > 0 else 'CC'
        brain["acumulation"][cond] = (((history[-1][opp_index] == 'C') if len(history) > 0 else True) - brain["tree"][cond])
        brain["total"][cond] = 1
        reg_lambda = 1
        Weta = {i: brain["acumulation"][i] / (brain["total"][i] + reg_lambda) for i in ('CC', 'CD', 'DC', 'DD')}
        brain["tree"]['C' + (history[-1][opp_index] if len(history) > 0 else 'C')] += Weta['C' + (history[-1][opp_index] if len(history) > 0 else 'C')]
        brain["tree"]['D' + (history[-1][opp_index] if len(history) > 0 else 'C')] += Weta['D' + (history[-1][opp_index] if len(history) > 0 else 'C')]
        E_c = brain["tree"]['C' + (history[-1][opp_index] if len(history) > 0 else 'C')]
        E_d = brain["tree"]['D' + (history[-1][opp_index] if len(history) > 0 else 'C')]
        if E_c >= E_d:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Egoist A-Learner':
        exploitation_rate = 2
        exploration_rate = 0.05
        if len(history) > 0:
            old_cond = internal_memory[228][1]
            reward = (score[self_index] - score[opp_index])
            internal_memory[228][0][old_cond][(history[-1][self_index] == 'D')] += reward * exploitation_rate
        cond = (history[-1][self_index] + history[-1][opp_index]) if len(history) > 0 else 'CC'
        internal_memory[228][1] = cond
        if (rnd.random() <= exploration_rate) or (internal_memory[228][0][cond][0] == internal_memory[228][0][cond][1]):
            return rnd.choice(history[-1]) if len(history) > 0 else 'C'
        if internal_memory[228][0][cond][0] > internal_memory[228][0][cond][1]:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Syifa':
        if not len(history):
            return 'C'
        if len(history) >= (tournament_avg_last_round):
            return 'D'
        if rnd.random() <= 0.1:
            if history[-1][opp_index] == 'C':
                internal_memory[229] = 1
        if internal_memory[229] < 0:
            internal_memory[229] += 1
            return 'C'
        if internal_memory[229] == 1:
            if history[-1][opp_index] == 'D':
                internal_memory[229] = -1
                return 'C'
            return 'D'
        else:
            return history[-1][opp_index]
    elif strategy == 'Trader':
        brain = internal_memory[230]
        trader_eyes = [('C', 'C') for _ in range(3)] + history
        if not len(history):
            RRR = brain["RRR"]
            if RRR < 1:
                brain["ALLD"] = True
                return 'D'
        if brain["ALLD"]:
            return 'D'
        if trader_eyes[-1][opp_index] == 'C':
            if brain["B"][1] == 1:
                brain["Lowest score ever get"] = min(brain["Lowest score ever get"], brain["B"][0])
                brain["B"][0] = 0
                brain["B"][1] = 0
            brain["Win round total"] += 1
            brain["Total poin opp C"] += score[self_index] - brain["Old score"]
            brain["G"][0] += score[self_index] - brain["Old score"]
            brain["G"][1] = 1
        else:
            if brain["G"][1] == 1:
                brain["Highest score ever get"] = max(brain["Highest score ever get"], brain["G"][0])
                brain["G"][0] = 0
                brain["G"][1] = 0
            brain["Lose round total"] += 1
            brain["Total poin opp D"] += score[self_index] - brain["Old score"]
            brain["B"][0] += score[self_index] - brain["Old score"]
            brain["B"][1] = 1
        if (len(history)) < 3:
            return 'C'
        if brain["Lose round total"] >= brain["Tolerant"]:
            brain["ALLD"] = True
            return 'D'
        brain["Old score"] = score[self_index]
        if (len(history) + 1) % 10 == 0:
            win_rate = brain["Win round total"] / (len(history) + 1)
            lose_rate = 1 - win_rate
            profit_factor = brain["Total poin opp C"] / brain["Total poin opp D"] if brain["Total poin opp D"] > 0 else 999.0
            expectation = (win_rate * ((brain["Total poin opp C"] / brain["Win round total"]) if brain["Win round total"] > 0 else 0)) - (lose_rate * ((brain["Total poin opp D"] / brain["Lose round total"]) if brain["Lose round total"] > 0 else 0))
            drawdown = (brain["Highest score ever get"] - brain["Lowest score ever get"]) / brain["Highest score ever get"]
            brain["Highest score ever get"] = self_avg_payoff_matrix
            brain["Lowest score ever get"] = self_avg_payoff_matrix
            if drawdown > 0.25 or profit_factor < 1 or expectation < 1:
                return 'D'
        inp = [(trader_eyes[-i][opp_index] == 'C') for i in range(1, 3 + 1)] 
        SMA = sum(inp) / 3
        pivot_point = (1 + 0 + (trader_eyes[-1][opp_index] == 'C')) / 3
        if SMA < 0.66 or pivot_point < 0.5:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Procastinater':
        if internal_memory[232] > 0:
            internal_memory[232] -= 1
            return 'D'
        if ((score[opp_index] - score[self_index]) >= (self_payoff_matrix['DC'] * 10)):
            internal_memory[232] = 10
        return 'C'
    elif strategy == 'Evolvable Cycler Attariq Version':
        min_length_gene = 1
        max_length_gene = 7
        def breeding_and_mutating(x, y):
            BAM = []
            length = rnd.randint(min(len(x), len(y)), max(len(x), len(y)))
            for i in range(length):
                if i > (len(x) - 1):
                    BAM.append(y[i])
                elif i > (len(y) - 1):
                    BAM.append(x[i])
                else:
                    BAM.append(rnd.choice([x[i], y[i]]))
            gene_choices = [1]
            if len(BAM) < max_length_gene:
                gene_choices.append(2)
            if len(BAM) > min_length_gene:
                gene_choices.append(3)
            R = rnd.choice(gene_choices)
            if R == 1:
                BAM[rnd.randint(0, len(BAM) - 1)] = rnd_prob(0.5)
            elif R == 2:
                BAM.insert(rnd.randint(0,len(BAM)), rnd_prob(0.5))
            elif R == 3:
                BAM.pop(rnd.randint(0,len(BAM)-1))
            return BAM
        brain = internal_memory[244]
        A = (len(history) + 1) - (brain["sum old index"] + 1)
        if len(history) > 0 and A == len(brain["now pattern"]):
            brain["pattern score"] += score[self_index] - brain["old score"]
            brain["old score"] = score[self_index]
            brain["sum old index"] += len(brain["now pattern"])
            if brain["pattern score"] > brain["best pattern 1"][1]:
                brain["best pattern 2"] = [brain["best pattern 1"][0][:], brain["best pattern 1"][1]]
                brain["best pattern 1"] = [brain["now pattern"][:], brain["pattern score"]]
            elif brain["pattern score"] > brain["best pattern 2"][1]:
                brain["best pattern 2"] = [brain["now pattern"][:], brain["pattern score"]]
            brain["now pattern"] = breeding_and_mutating(brain["best pattern 1"][0], brain["best pattern 2"][0])
            brain["pattern score"] = 0 
            A = (len(history) + 1) - (brain["sum old index"] + 1)
        if A < len(brain["now pattern"]):
            brain["pattern score"] += score[self_index] - brain["old score"]
            brain["old score"] = score[self_index]
            return brain["now pattern"][A]
    elif strategy == 'Tit For Tat Killer':
        if len(history) <= 2:
            return ('D', 'C', 'C')[len(history)]
        return 'D' if [history[1][opp_index], history[2][opp_index]] == ['D', 'C'] else 'C'
    elif strategy == 'ZD-2026 / Zero Determinant 2026':
        # == BASES == 
        phi = 1.0
        s = 0.0
        l = 0.0
        # == BIASES ==
        if len(history) > 1:
            if history[-2][self_index] == 'C':
                internal_memory[313][1] += 1
                if history[-1][opp_index] == 'D':
                    internal_memory[313][0] += 1
        extort_prob = max(min((internal_memory[313][0] / max(internal_memory[313][1], 1)) * self_payoff_matrix['DC'], 1), 0)
        phi = phi + 0
        s = s + (1 - extort_prob)
        l = l + ((self_payoff_matrix['DD'] * extort_prob) + (self_payoff_matrix['CC'] * (1 - extort_prob)))
        # == CLAMPS ==
        clamped_l = max(min(l, self_payoff_matrix['CC']), self_payoff_matrix['DD'])
        s_min = -min((self_payoff_matrix['DC'] - clamped_l) / (clamped_l - self_payoff_matrix['CD']), (clamped_l - self_payoff_matrix['CD']) / (self_payoff_matrix['DC'] - clamped_l))
        phi = max(min(phi, 1), 0)
        s = max(min(s, 1), s_min)
        l = clamped_l
        # == ZERO DETERMINANT COMPONENT ==
        p1 = 1 - phi * (1 - s) * (self_payoff_matrix['CC'] - l)
        p2 = 1 - phi * (s * (l - self_payoff_matrix['CD']) + (self_payoff_matrix['DC'] - l))
        p3 = phi * ((l - self_payoff_matrix['CD']) + s * (self_payoff_matrix['DC'] - l))
        p4 = phi * (1 - s) * (l - self_payoff_matrix['DD'])
        # == OUTPUT ==
        four_vector = [p1, p2, p3, p4]
        return 'C' if rnd.random() <= four_vector[((2 * (history[-1][self_index] == 'D')) + (history[-1][opp_index] == 'D')) if len(history) > 0 else 0] else 'D'
    elif strategy == 'ZD-PW2026 / Zero Determinant Psychological War 2026':
        # == BASES == 
        phi = 0.8
        s = 0.15
        l = 0.2
        # == BIASES ==
        if len(history) > 1:
            if history[-2][self_index] == 'C':
                internal_memory[317][1] += 1
                if history[-1][opp_index] == 'D':
                    internal_memory[317][0] += 1
        extort_prob = max(min((internal_memory[317][0] / max(internal_memory[317][1], 1)) * self_payoff_matrix['DC'], 1), 0)
        phi = phi + 0
        s = s + ((1 - extort_prob) * 0.7)
        l = l + ((self_payoff_matrix['DD'] * extort_prob) + (self_payoff_matrix['CC'] * (1 - extort_prob)))
        # == CLAMPS ==
        clamped_l = max(min(l, self_payoff_matrix['CC']), self_payoff_matrix['DD'])
        s_min = -min((self_payoff_matrix['DC'] - clamped_l) / (clamped_l - self_payoff_matrix['CD']), (clamped_l - self_payoff_matrix['CD']) / (self_payoff_matrix['DC'] - clamped_l))
        phi = max(min(phi, 1), 0)
        s = max(min(s, 1), s_min)
        l = clamped_l
        # == ZERO DETERMINANT COMPONENT ==
        p1 = 1 - phi * (1 - s) * (self_payoff_matrix['CC'] - l)
        p2 = 1 - phi * (s * (l - self_payoff_matrix['CD']) + (self_payoff_matrix['DC'] - l))
        p3 = phi * ((l - self_payoff_matrix['CD']) + s * (self_payoff_matrix['DC'] - l))
        p4 = phi * (1 - s) * (l - self_payoff_matrix['DD'])
        # == OUTPUT ==
        four_vector = [p1, p2, p3, p4]
        return 'C' if rnd.random() <= four_vector[((2 * (history[-1][self_index] == 'D')) + (history[-1][opp_index] == 'D')) if len(history) > 0 else 0] else 'D'
    elif strategy == 'Sort':
        if not len(history):
            return 'C'
        B = min(10, len(history))
        inp = [history[-i][opp_index] for i in range(B, 0, -1)]
        if sorted(inp) == inp:
            return 'C'
        return 'D'
    elif strategy == 'Backrooms':
        decay_rate = 0.05
        internal_memory[337][1] = {i: ((internal_memory[337][1][i] * (1 - decay_rate)) + (rnd.random() * decay_rate)) for i in internal_memory[337][1]}
        if len(history) % 25 == 0 and len(history) > 0:
            new_dict = {}
            total = {}
            for i in range(24, 0, -1):
                if new_dict.get(history[-(i + 1)][self_index] + history[-(i + 1)][opp_index]) == None: 
                    new_dict[history[-(i + 1)][self_index] + history[-(i + 1)][opp_index]] = 0 
                    total[history[-(i + 1)][self_index] + history[-(i + 1)][opp_index]] = 0 
                new_dict[history[-(i + 1)][self_index] + history[-(i + 1)][opp_index]] += (history[-i][opp_index] == 'C') 
                total[history[-(i + 1)][self_index] + history[-(i + 1)][opp_index]] += 1 
            internal_memory[337][0] = {i: ((new_dict[i] / total[i]) if i in new_dict else ((internal_memory[337][0][i] * (1 - decay_rate)) + (rnd.random() * decay_rate))) for i in ['CC', 'CD', 'DC', 'DD']}
            internal_memory[337][1] = {i: internal_memory[337][0][i] for i in internal_memory[337][0]}
        internal_memory[337][1] = {i: ((internal_memory[337][1][i] * (1 - decay_rate)) + (rnd.random() * decay_rate)) for i in internal_memory[337][1]}
        return 'C' if rnd.random() <= internal_memory[337][1][(history[-1][self_index] + history[-1][opp_index]) if len(history) > 0 else 'CC'] else 'D'
    elif strategy == 'Always Out / Outtor':
        return '[OUT]'
    elif strategy == 'Out For Tat':
        if not len(history):
            return 'C'
        return '[OUT]' if history[-1][opp_index] == 'D' else 'C'
    elif strategy == 'Bandit':
        learning_rate = 0.1
        epsilon = 0.1
        Q_table = internal_memory[339][0]
        if len(history) > 0:
            old_score = internal_memory[339][1]
            Q_old = Q_table[history[-2][opp_index] if len(history) > 1 else 'C'][history[-1][self_index]]
            Q_new = Q_old + learning_rate * ((old_score - score[self_index]) - Q_old)
            internal_memory[339][1] = score[self_index]
        if rnd.random() <= epsilon:
            return rnd_prob(0.5)
        E_c = Q_table[history[-1][opp_index] if len(history) > 0 else 'C']['C']
        E_d = Q_table[history[-1][opp_index] if len(history) > 0 else 'C']['D']
        if E_c == E_d:
            return rnd_prob(0.5)
        if E_c > E_d:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Inequity Averse Q-Learner':
        brain = internal_memory[340][0]
        another_factor = internal_memory[340][1]
        A = min(12, len(history)) - 1
        opp_inp = tuple(['C' for _ in range(12 - min(12, len(history)))] + [history[-i][opp_index] for i in range(min(12, len(history)), 0, -1)]) 
        self_inp = tuple(['C' for _ in range(12 - min(12, len(history)))] + [history[-i][self_index] for i in range(min(12, len(history)), 0, -1)]) 
        inp = (opp_inp, self_inp)
        if brain.get(inp) == None:
            brain[inp] = [0,0]
        if len(history) > 0 and another_factor[0][1] != 0.5:
            reward = 0
            if history[-1][opp_index] == 'C':
                if history[-1][self_index] == 'C':
                    reward = self_payoff_matrix['CC']
                    opp_reward = self_payoff_matrix['CC']
                else:
                    reward = self_payoff_matrix['DC']
                    opp_reward = self_payoff_matrix['CD']
            else:
                if history[-1][self_index] == 'C':
                    reward = self_payoff_matrix['CD']
                    opp_reward = self_payoff_matrix['DC']
                else:
                    reward = self_payoff_matrix['DD']
                    opp_reward = self_payoff_matrix['DD']
            c_factor = brain[inp][0]
            d_factor = brain[inp][1]
            alpha = 0.5
            beta = 0.5
            utility = reward - ((alpha * max(0, reward - opp_reward)) - (beta * max(0, opp_reward - reward)))
            a = 0.9
            gamma = 0.9
            Q_old = brain[another_factor[0][0]][another_factor[0][1]]
            Q_new = Q_old + (a * (utility + ((gamma * max(c_factor, d_factor)) - Q_old)))
            brain[another_factor[0][0]][another_factor[0][1]] = Q_new
        if rnd.random() <= another_factor[1]:
            Action = rnd_prob(0.5)
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        decay = 1
        min_epsilon = 0
        another_factor[1] = max(another_factor[1] * decay, min_epsilon)
        if brain[inp][0] == brain[inp][1]:
            Action = rnd_prob(0.5)
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        best_choice = max(brain[inp])
        if best_choice == brain[inp][0]:
            another_factor[0] = [inp, 0]
            return 'C'
        else:
            another_factor[0] = [inp, 1]
            return 'D'
    elif strategy == 'Sarsa Q-Learner':
        brain = internal_memory[341][0]
        another_factor = internal_memory[341][1]
        A = min(12, len(history)) - 1
        opp_inp = tuple(['C' for _ in range(12 - min(12, len(history)))] + [history[-i][opp_index] for i in range(min(12, len(history)), 0, -1)]) 
        self_inp = tuple(['C' for _ in range(12 - min(12, len(history)))] + [history[-i][self_index] for i in range(min(12, len(history)), 0, -1)]) 
        inp = (opp_inp, self_inp)
        if brain.get(inp) == None:
            brain[inp] = [0,0]
        if len(history) > 0 and another_factor[0][1] != 0.5:
            reward = 0
            if history[-1][opp_index] == 'C':
                if history[-1][self_index] == 'C':
                    reward = self_payoff_matrix['CC']
                else:
                    reward = self_payoff_matrix['DC']
            else:
                if history[-1][self_index] == 'C':
                    reward = self_payoff_matrix['CD']
                else:
                    reward = self_payoff_matrix['DD']
            c_factor = brain[inp][0]
            d_factor = brain[inp][1]
            a = 0.9
            gamma = 0.9
            Q_old = brain[another_factor[0][0]][another_factor[0][1]]
            Q_new = Q_old + (a * (reward + ((gamma * [c_factor, d_factor][(history[-1][self_index] == 'D')]) - Q_old)))
            brain[another_factor[0][0]][another_factor[0][1]] = Q_new
        if rnd.random() <= another_factor[1]:
            Action = rnd_prob(0.5)
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        decay = 1
        min_epsilon = 0
        another_factor[1] = max(another_factor[1] * decay, min_epsilon)
        if brain[inp][0] == brain[inp][1]:
            Action = rnd_prob(0.5)
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        best_choice = max(brain[inp])
        if best_choice == brain[inp][0]:
            another_factor[0] = [inp, 0]
            return 'C'
        else:
            another_factor[0] = [inp, 1]
            return 'D'
    elif strategy == 'REINFORCE':
        brain = internal_memory[342]
        learning_rate = 0.5
        if len(history) > 0:
            brain[brain[2]] += learning_rate * (self_payoff_matrix[history[-1][self_index] + history[-1][opp_index]] - self_avg_payoff_matrix)
        total = mth.exp(brain[0]) + mth.exp(brain[1])
        E_c = mth.exp(brain[0]) / total
        E_d = 1 - E_c
        if E_c == E_d:
            Action = rnd_prob(0.5)
            brain[2] = (Action == 'D')
            return Action
        if E_c > E_d:
            brain[2] = 0
            return 'C'
        else:
            brain[2] = 1
            return 'D'
    elif strategy == 'Dyna Q-Learner':
        brain = internal_memory[343][0]
        another_factor = internal_memory[343][1]
        A = min(12, len(history)) - 1
        opp_inp = tuple(['C' for _ in range(12 - min(12, len(history)))] + [history[-i][opp_index] for i in range(min(12, len(history)), 0, -1)]) 
        self_inp = tuple(['C' for _ in range(12 - min(12, len(history)))] + [history[-i][self_index] for i in range(min(12, len(history)), 0, -1)]) 
        inp = (opp_inp, self_inp)
        if brain.get(inp) == None:
            brain[inp] = [0,0]
        if len(history) > 0 and another_factor[0][1] != 0.5:
            reward = 0
            if history[-1][opp_index] == 'C':
                if history[-1][self_index] == 'C':
                    reward = self_payoff_matrix['CC']
                else:
                    reward = self_payoff_matrix['DC']
            else:
                if history[-1][self_index] == 'C':
                    reward = self_payoff_matrix['CD']
                else:
                    reward = self_payoff_matrix['DD']
            a = 0.9
            gamma = 0.9
            c_factor = brain[inp][0]
            d_factor = brain[inp][1]
            Q_old = brain[another_factor[0][0]][another_factor[0][1]]
            Q_new = Q_old + (a * (reward + ((gamma * max(c_factor, d_factor)) - Q_old)))
            for i in range(5):
                test = rnd.choice([i for i in brain])
                reward = 0
                if test[opp_index][-1] == 'C':
                    if test[self_index][-1] == 'C':
                        reward = self_payoff_matrix['CC']
                    else:
                        reward = self_payoff_matrix['DC']
                else:
                    if test[self_index][-1] == 'C':
                        reward = self_payoff_matrix['CD']
                    else:
                        reward = self_payoff_matrix['DD']
                c_factor = brain[inp][0]
                d_factor = brain[inp][1]
                Q_new += (a * (reward + ((gamma * max(c_factor, d_factor)) - Q_new)))
            brain[another_factor[0][0]][another_factor[0][1]] = Q_new
        if rnd.random() <= another_factor[1]:
            Action = rnd_prob(0.5)
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        decay = 1
        min_epsilon = 0
        another_factor[1] = max(another_factor[1] * decay, min_epsilon)
        if brain[inp][0] == brain[inp][1]:
            Action = rnd_prob(0.5)
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        best_choice = max(brain[inp])
        if best_choice == brain[inp][0]:
            another_factor[0] = [inp, 0]
            return 'C'
        else:
            another_factor[0] = [inp, 1]
            return 'D'
    elif strategy == 'Continuous Q-Learner Avg':
        brain = internal_memory[344]
        reward = self_avg_payoff_matrix
        if len(history) > 0:
            learning_rate = 0.1
            brain[0] = ((1 - learning_rate) * brain[0]) + (learning_rate * (history[-1][opp_index] == 'C'))
            brain[1] = ((1 - learning_rate) * brain[1]) + (learning_rate * (history[-1][opp_index] == 'D'))
            reward = self_payoff_matrix[history[-1][self_index] + history[-1][opp_index]]
        X_sh = (brain[0] * self_payoff_matrix['CC']) + (brain[1] * self_payoff_matrix['CD'])
        X_st = (brain[0] * self_payoff_matrix['DC']) + (brain[1] * self_payoff_matrix['DD'])
        Action = reward - ((X_sh + X_st) / 2)
        kappa = 2.0
        A = 1 / (1 + mth.exp(-Action * kappa))
        if rnd.random() <= A:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Q-Learning A-Learner':
        exploitation_rate = 0.9
        exploration_rate = 0.1
        if len(history) > 0:
            gamma = 0.9
            c_factor = sum(internal_memory[345][0][i][0] for i in internal_memory[345][0]) / len(internal_memory[345][0])
            d_factor = sum(internal_memory[345][0][i][1] for i in internal_memory[345][0]) / len(internal_memory[345][0])
            old_cond = internal_memory[345][1]
            old_score = internal_memory[345][2]
            reward = (score[self_index] - old_score)
            internal_memory[345][0][old_cond][(history[-1][self_index] == 'D')] += (exploitation_rate * (reward + ((gamma * max(c_factor, d_factor)) - internal_memory[345][0][old_cond][(history[-1][self_index] == 'D')])))
            internal_memory[345][2] = score[self_index]
        cond = (history[-1][self_index] + history[-1][opp_index]) if len(history) > 0 else 'CC'
        internal_memory[345][1] = cond
        if (rnd.random() <= exploration_rate) or (internal_memory[345][0][cond][0] == internal_memory[345][0][cond][1]):
            return rnd.choice(history[-1]) if len(history) > 0 else 'C'
        if internal_memory[345][0][cond][0] > internal_memory[345][0][cond][1]:
            return 'C'
        else:
            return 'D'
    elif strategy == 'A-A&C / A-Actor&Critic':
        brain = internal_memory[346]
        if len(history) > 0:
            learning_rate = 0.5
            brain[(history[-2][self_index] + history[-2][opp_index]) if len(history) > 1 else 'CC'][(history[-1][self_index] == 'D')] += learning_rate * (self_payoff_matrix[(history[-1][self_index] + history[-1][opp_index])] - (score[self_index] / len(history)))
        pref = brain[(history[-1][self_index] + history[-1][opp_index]) if len(history) > 0 else 'CC']
        return rnd_prob(1 / (1 + mth.exp(pref[1] - pref[0])))
    elif strategy == 'Japan Empire Emperor':
        if not len(history):
            next_chat[0] = '私はお前の皇帝だ!'
            return 'C'
        elif len(history) == 1:
            if chats[opp_index] == '私はあなたの兵士だ!':
                internal_memory[347] = True
        if internal_memory[347]:
            return 'D'
        return 'C' if score[self_index] >= score[opp_index] else 'D'
    elif strategy == 'Japan Empire Soldier 1':
        if not len(history):
            next_chat[0] = '私はあなたの兵士だ!'
            return 'C'
        elif len(history) == 1:
            if chats[opp_index] in ('私はお前の皇帝だ!', '私はあなたの兵士だ!'):
                internal_memory[348] = True
        if internal_memory[348]:
            return 'C'
        return '[OUT]'
    elif strategy == 'Japan Empire Soldier 2':
        if not len(history):
            next_chat[0] = '私はあなたの兵士だ!'
            return 'C'
        elif len(history) == 1:
            if chats[opp_index] in ('私はお前の皇帝だ!', '私はあなたの兵士だ!'):
                internal_memory[349] = True
        if internal_memory[349]:
            return 'C'
        return '[OUT]'
    elif strategy == 'Japan Empire Soldier 3':
        if not len(history):
            next_chat[0] = '私はあなたの兵士だ!'
            return 'C'
        elif len(history) == 1:
            if chats[opp_index] in ('私はお前の皇帝だ!', '私はあなたの兵士だ!'):
                internal_memory[350] = True
        if internal_memory[350]:
            return 'C'
        return '[OUT]'
    elif strategy == 'Japan Empire Soldier 4':
        if not len(history):
            next_chat[0] = '私はあなたの兵士だ!'
            return 'C'
        elif len(history) == 1:
            if chats[opp_index] in ('私はお前の皇帝だ!', '私はあなたの兵士だ!'):
                internal_memory[351] = True
        if internal_memory[351]:
            return 'C'
        return '[OUT]'
    elif strategy == 'Japan Empire Soldier 5':
        if not len(history):
            next_chat[0] = '私はあなたの兵士だ!'
            return 'C'
        elif len(history) == 1:
            if chats[opp_index] in ('私はお前の皇帝だ!', '私はあなたの兵士だ!'):
                internal_memory[352] = True
        if internal_memory[352]:
            return 'C'
        return '[OUT]'
    elif strategy == 'Japan Empire Soldier 6':
        if not len(history):
            next_chat[0] = '私はあなたの兵士だ!'
            return 'C'
        elif len(history) == 1:
            if chats[opp_index] in ('私はお前の皇帝だ!', '私はあなたの兵士だ!'):
                internal_memory[353] = True
        if internal_memory[353]:
            return 'C'
        return '[OUT]'
    elif strategy == 'Japan Empire Soldier 7':
        if not len(history):
            next_chat[0] = '私はあなたの兵士だ!'
            return 'C'
        elif len(history) == 1:
            if chats[opp_index] in ('私はお前の皇帝だ!', '私はあなたの兵士だ!'):
                internal_memory[354] = True
        if internal_memory[354]:
            return 'C'
        return '[OUT]'
    elif strategy == 'Japan Empire Soldier 8':
        if not len(history):
            next_chat[0] = '私はあなたの兵士だ!'
            return 'C'
        elif len(history) == 1:
            if chats[opp_index] in ('私はお前の皇帝だ!', '私はあなたの兵士だ!'):
                internal_memory[355] = True
        if internal_memory[355]:
            return 'C'
        return '[OUT]'
    elif strategy == 'Japan Empire Soldier 9':
        if not len(history):
            next_chat[0] = '私はあなたの兵士だ!'
            return 'C'
        elif len(history) == 1:
            if chats[opp_index] in ('私はお前の皇帝だ!', '私はあなたの兵士だ!'):
                internal_memory[356] = True
        if internal_memory[356]:
            return 'C'
        return '[OUT]'
    elif strategy == 'Japan Empire Soldier 10':
        if not len(history):
            next_chat[0] = '私はあなたの兵士だ!'
            return 'C'
        elif len(history) == 1:
            if chats[opp_index] in ('私はお前の皇帝だ!', '私はあなたの兵士だ!'):
                internal_memory[357] = True
        if internal_memory[357]:
            return 'C'
        return '[OUT]'
    elif strategy == 'Prabowo':
        if not len(history):
            return 'C'
        if internal_memory[358] == 4:
            if history[-1][opp_index] == 'D':
                internal_memory[358] = 0
                next_chat[0] = 'SAWIT KAN POHON JUGA!!!'
            else:
                next_chat[0] = 'AKU SUKA SAWIT!!!'
            return 'D'
        if 1 <= internal_memory[358] <= 3:
            if internal_memory[358] > 1:
                internal_memory[358] -= 1
                next_chat[0] = 'MBG!!!'
                return ('D', 'C')[2 - internal_memory[358]]
            if history[-1][opp_index] == 'D':
                internal_memory[358] = 0
                next_chat[0] = 'SAYA KAN KADANG-KADANG SALAH MAKAN JUGA!!!'
        if rnd.random() <= 0.2:
            internal_memory[358] = 3 if rnd.random() <= 0.75 else 4
        if history[-1][opp_index] == 'D':
            next_chat[0] = 'HEY, ANTEK-ANTEK ASING!!!'
            return 'D'
        return 'C'
    elif strategy == 'Gen 5_64_200_30_0.3_A':
        def sigmoid10(x):
            return 1 / (1 + mth.exp(-max(min(x * 10, 10), -10)))
        return rnd_prob(sigmoid10(((((history[-3][opp_index] == 'C') if len(history) >= 3 else 0.5) * ((history[-3][self_index] == 'C') if len(history) >= 3 else 0.5)) - ((((((history[-1][self_index] == 'C') if len(history) >= 1 else 0.5) / (((history[-10][self_index] == 'C') if len(history) >= 10 else 0.5) if ((history[-10][self_index] == 'C') if len(history) >= 10 else 0.5) != 0 else 1.0)) if ((history[-1][opp_index] == 'C') if len(history) >= 1 else 0.5) > 0 else ((history[-8][self_index] == 'C') if len(history) >= 8 else 0.5)) / (((history[-2][self_index] == 'C') if len(history) >= 2 else 0.5) if ((history[-2][self_index] == 'C') if len(history) >= 2 else 0.5) != 0 else 1.0)) / (0.0 if 0.0 != 0 else 1.0)))))
    elif strategy == 'Gen 3_64_200_30_0.3_A':
        def sigmoid10(x):
            return 1 / (1 + mth.exp(-max(min(x * 10, 10), -10)))
        return rnd_prob(sigmoid10((0.0 - ((((history[-10][opp_index] == 'C') if len(history) >= 10 else 0.5) + ((history[-1][opp_index] == 'C') if len(history) >= 1 else 0.5)) / (((history[-8][self_index] == 'C') if len(history) >= 8 else 0.5) if ((history[-8][self_index] == 'C') if len(history) >= 8 else 0.5) != 0 else 1.0)))))
    elif strategy == 'Gen 4_64_200_30_0.3_A':
        def sigmoid10(x):
            return 1 / (1 + mth.exp(-max(min(x * 10, 10), -10)))
        return rnd_prob(sigmoid10(((((((history[-1][opp_index] == 'C') if len(history) >= 1 else 0.5) - ((history[-1][self_index] == 'C') if len(history) >= 1 else 0.5)) * ((history[-8][self_index] == 'C') if len(history) >= 8 else 0.5)) - ((((history[-7][self_index] == 'C') if len(history) >= 7 else 0.5) * ((history[-9][opp_index] == 'C') if len(history) >= 9 else 0.5)) / ((1.0 * ((history[-7][self_index] == 'C') if len(history) >= 7 else 0.5)) if (1.0 * ((history[-7][self_index] == 'C') if len(history) >= 7 else 0.5)) != 0 else 1.0))) - ((history[-1][opp_index] == 'C') if len(history) >= 1 else 0.5))))
    elif strategy == 'Gen 2_64_200_30_0.3_A':
        def sigmoid10(x):
            return 1 / (1 + mth.exp(-max(min(x * 10, 10), -10)))
        return rnd_prob(sigmoid10(((((history[-7][self_index] == 'C') if len(history) >= 7 else 0.5) if 1.0 > 0 else ((history[-8][self_index] == 'C') if len(history) >= 8 else 0.5)) - (((history[-2][opp_index] == 'C') if len(history) >= 2 else 0.5) / (((history[-1][self_index] == 'C') if len(history) >= 1 else 0.5) if ((history[-1][self_index] == 'C') if len(history) >= 1 else 0.5) != 0 else 1.0)))))
    elif strategy == 'Gen 1_64_200_30_0.3_A':
        def sigmoid10(x):
            return 1 / (1 + mth.exp(-max(min(x * 10, 10), -10)))
        return rnd_prob(sigmoid10((((history[-1][self_index] == 'C') if len(history) >= 1 else 0.5) - 1.0)))
    elif strategy == 'Bahlil':
        if not len(history):
            return 'C'
        if internal_memory[359]:
            if history[-1][opp_index] == 'C':
                next_chat[0] = 'BENSIN OPLOSAN!!!'
                return 'D'
            else:
                internal_memory[359] = False
                next_chat[0] = 'itu kan namanya dicampur ethanol bukan di oploskan ):-]'
                return 'C'
        if rnd.random() <= 0.2:
            internal_memory[359] = True
        if history[-1][opp_index] == 'D':
            next_chat[0] = ':-]'
            return 'D'
        return 'C'
    elif strategy == 'Pawlow':
        if not len(history):
            return 'C'
        if (score[self_index] / len(history)) >= self_payoff_matrix['CC']:
            return history[-1][self_index]
        return flip(history[-1][self_index])
    elif strategy == 'Rememberable Grudger':
        if immortal_memory[3].get(opp_name) != None:
            if len(history) > 0:
                if history[-1][opp_index] == 'D':
                    immortal_memory[3][opp_name] = min(len(history) - 1, immortal_memory[3][opp_name])
            if len(history) >= immortal_memory[3][opp_name]:
                return 'D'
        else:
            if len(history) > 0:
                if history[-1][opp_index] == 'D':
                    immortal_memory[3][opp_name] = len(history) - 1
                    return 'D'
        return 'C'
    elif strategy == 'Collatz':
        if not len(history):
            internal_memory[269][2] = internal_memory[269][1]
        if internal_memory[269][2] == 0:
            internal_memory[269][0] = flip(internal_memory[269][0])
            internal_memory[269][1] = (internal_memory[269][1] / 2) if internal_memory[269][1] % 2 == 0 else ((internal_memory[269][1] * 3) + 1)
            internal_memory[269][2] = internal_memory[269][1]
        internal_memory[269][2] -= 1
        return internal_memory[269][0]
    elif strategy == 'Odysseus / Ὀδυσσεύς':
        if not len(history):
            return 'C'
        if (score[self_index] - score[opp_index]) >= (2 * self_payoff_matrix['DC']):
            next_chat[0] = f"{opp_name}! Εἴ τίς σε ἀνθρώπων ἔρηται περὶ τῆς αἰσχρᾶς σῆς τῶν βαθμῶν φθορᾶς, φάσθαι Ὀδυσσῆα πτολίπορθον ἐξαλαῶσαί σου τὸν ἀριθμόν, τὸν Λαερτιάδην, οὗ οἶκος ἐν ὑψίστοις πίναξιν ἔσται!"
        if (history[-1][self_index], history[-1][opp_index]) == ('C', 'D'):
            if immortal_memory[4].get(opp_name) == None:
                immortal_memory[4][opp_name] = 0
            immortal_memory[4][opp_name] += 1
        A = frequency(history[-1][self_index], 'D', history, opp_index)
        if None not in (A, immortal_memory[4].get(opp_name)):
            return rnd_prob(max(min(1 - (A * (1 + (immortal_memory[4][opp_name] * 0.05))), 1), 0))
        return 'C'
    elif strategy == 'Newton':
        if len(history) > 0:
            if history[-1][opp_index] == 'D':
                internal_memory[384] += 1
        m = (internal_memory[384] / len(history)) if len(history) > 0 else 0.5
        a = (min(len(history), 10) / ((min(len(history), 10) * 0.5) * (min(len(history), 10) * 0.5))) if len(history) > 0 else 1
        f = m * a
        return rnd_prob(1 - (min(max(f, 0), 1)))
    elif strategy == 'Coin':
        if not len(history):
            return rnd_prob(0.5)
        return history[-1][self_index] if rnd.random() <= 0.51 else flip(history[-1][self_index])
    elif strategy == 'LCG':
        internal_memory[385][3] = ((internal_memory[385][0] * internal_memory[385][3]) + internal_memory[385][1]) % internal_memory[385][2]
        return 'DC'[(internal_memory[385][3] >= (internal_memory[385][2] / 2))]
    elif strategy == 'Your Choice Or My Choice':
        if not len(history):
            return 'C'
        return rnd.choice(history[-1])
    elif strategy == 'L-BODOR / Learning Based On Derivative Of Reward':
        if len(history) > 0:
            learning_rate = 0.25
            diff = 0.75
            reward = (score[self_index] - (internal_memory[386][0] + self_avg_payoff_matrix)) / diff
            internal_memory[386][0] = score[self_index]
            A1 = internal_memory[386][1][(history[-2][self_index] + history[-2][opp_index]) if len(history) > 1 else 'CC'][(history[-1][self_index] == 'D')]
            internal_memory[386][1][(history[-2][self_index] + history[-2][opp_index]) if len(history) > 1 else 'CC'][(history[-1][self_index] == 'D')] = ((1 - learning_rate) * A1) + (learning_rate * reward)
        A2 = internal_memory[386][1][(history[-1][self_index] + history[-1][opp_index]) if len(history) > 0 else 'CC']
        return 'CD'[A2.index(max(A2))]
    elif strategy == 'TOM Level 1 / Theory Of Mind Level 1':
        models = {
            'ALLC': lambda my_h, opp_h: 'C',
            'ALLD': lambda my_h, opp_h: 'D',
            'TFT':  lambda my_h, opp_h: my_h[-1] if my_h else 'C',
            'GRIM': lambda my_h, opp_h: 'D' if 'D' in my_h else 'C'
        }
        if not len(history):
            internal_memory[404] = {m: 0 for m in models}
            return 'C'
        my_history = [m[self_index] for m in history]
        opp_history = [m[opp_index] for m in history]
        last_opp_move = opp_history[-1]
        prev_my_h = my_history[:-1]
        prev_opp_h = opp_history[:-1]
        if prev_my_h:
            for name, model_func in models.items():
                predicted = model_func(prev_my_h, prev_opp_h)
                if predicted == last_opp_move:
                    internal_memory[404][name] += 1
        best_model_name = max(internal_memory[404], key=internal_memory[404].get)
        predicted_next_opp_move = models[best_model_name](my_history, opp_history)
        if best_model_name == 'ALLC':
            return 'D'
        elif predicted_next_opp_move == 'D':
            return 'D'
        else:
            return 'C'
    elif strategy == 'TOM Level 2 / Theory Of Mind Level 2':
        if len(history) < 2:
            return 'C'
        my_history = [m[self_index] for m in history]
        opp_history = [m[opp_index] for m in history]
        my_c_ratio = my_history.count('C') / len(history)
        predicted_opp_level1_think = 'D' if (my_c_ratio > 0.7 or opp_history[-1] == 'D') else 'C'
        if predicted_opp_level1_think == 'D':
            return 'D'
        else:
            return 'C'
    elif strategy == 'TOM Level 3 / Theory Of Mind Level 3':
        if len(history) < 3:
            return 'C'
        my_history = [m[self_index] for m in history]
        opp_history = [m[opp_index] for m in history]
        my_c_ratio = my_history.count('C') / len(history)
        opp_last_move = opp_history[-1]
        opp_prev_move = opp_history[-2]
        is_opp_trapping = (opp_prev_move == 'C' and opp_last_move == 'D' and my_history[-1] == 'C')
        if is_opp_trapping:
            return 'D'
        if 0.65 <= my_c_ratio <= 0.75 and my_history[-1] == 'C':
            return 'D'
        return 'C' if opp_last_move == 'C' else 'D'
    elif strategy == 'TPMS / Tree Predictive Matrix Strategy':
        if not len(history):
            return 'C'
        if len(history) == 1:
            return history[-1][opp_index]
        level = 5
        num_samples = 25
        totals = [
            sum([((history[-i][self_index], history[-i][opp_index]) == ('C', 'C')) for i in range(min(num_samples, len(history)), 1, -1)]),
            sum([((history[-i][self_index], history[-i][opp_index]) == ('C', 'D')) for i in range(min(num_samples, len(history)), 1, -1)]),
            sum([((history[-i][self_index], history[-i][opp_index]) == ('D', 'C')) for i in range(min(num_samples, len(history)), 1, -1)]),
            sum([((history[-i][self_index], history[-i][opp_index]) == ('D', 'D')) for i in range(min(num_samples, len(history)), 1, -1)]),
        ]
        c_responses = [
            sum([((history[-i][self_index], history[-i][opp_index], history[-i + 1][opp_index]) == ('C', 'C', 'C')) for i in range(min(num_samples, len(history)), 1, -1)]),
            sum([((history[-i][self_index], history[-i][opp_index], history[-i + 1][opp_index]) == ('C', 'D', 'C')) for i in range(min(num_samples, len(history)), 1, -1)]),
            sum([((history[-i][self_index], history[-i][opp_index], history[-i + 1][opp_index]) == ('D', 'C', 'C')) for i in range(min(num_samples, len(history)), 1, -1)]),
            sum([((history[-i][self_index], history[-i][opp_index], history[-i + 1][opp_index]) == ('D', 'D', 'C')) for i in range(min(num_samples, len(history)), 1, -1)]),
        ]
        p_cc = (c_responses[0] / totals[0]) if totals[0] > 0 else 0.5
        p_cd = (c_responses[1] / totals[1]) if totals[1] > 0 else 0.5
        p_dc = (c_responses[2] / totals[2]) if totals[2] > 0 else 0.5
        p_dd = (c_responses[3] / totals[3]) if totals[3] > 0 else 0.5
        payoff = {
            ('C', 'C'): self_payoff_matrix['CC'], ('C', 'D'): self_payoff_matrix['CD'],
            ('D', 'C'): self_payoff_matrix['DC'], ('D', 'D'): self_payoff_matrix['DD']
        }
        prob = {
            ('C', 'C'): p_cc, ('C', 'D'): p_cd,
            ('D', 'C'): p_dc, ('D', 'D'): p_dd
        }
        def TPM(self_action, opp_action, depth):
            if depth == 0:
                return [prob[(self_action, opp_action)], payoff[(self_action, opp_action)]]
            TPM_val = [TPM('C', 'C', depth - 1), TPM('C', 'D', depth - 1), TPM('D', 'C', depth - 1), TPM('D', 'D', depth - 1)]
            max_val = [self_payoff_matrix['CD'], -1]
            for i in TPM_val:
                if (i[0] * i[1]) > max_val[0]:
                    max_val[0] = (i[0] * i[1])
                    max_val[1] += 1
            return [prob[(self_action, opp_action)] * TPM_val[max_val[1]][0], payoff[(self_action, opp_action)] + TPM_val[max_val[1]][1]]
        A = [TPM('C', 'C', level), TPM('C', 'D', level), TPM('D', 'C', level), TPM('D', 'D', level)]
        E_c = max(A[0][0] * A[0][1], A[1][0] * A[1][1])
        E_d = max(A[2][0] * A[2][1], A[3][0] * A[3][1])
        if E_c == E_d:
            return 'D' if 'D' in history[-1] else 'C'
        if E_c > E_d:
            return 'C'
        return 'D'
    elif strategy == 'Mystical Logic':
        if not len(history):
            return 'C'
        if internal_memory[408] > 1 and (history[-2][opp_index], history[-1][opp_index]) == ('C', 'C'):
            internal_memory[408] = 0
        if internal_memory[408] == 3 and history[-1][opp_index] == 'D':
            next_chat[0] = rnd.choice(['YOU ROBBER!😡', 'YOU JEALOUS!😡'])
        if internal_memory[408] == 2 and history[-1][opp_index] == 'D':
            next_chat[0] = 'HEY! DON\'T YOU DARE TAKE MY GIFT FROM THE SUPERNATURAL WORLD!😡'
            internal_memory[408] = 3
        if internal_memory[408] == 1 and history[-1][opp_index] == 'C':
            next_chat[0] = 'This is a gift from the supernatural world😏'
            internal_memory[408] = 2
        if internal_memory[408] == 0 and history[-1][opp_index] == 'D':
            next_chat[0] = rnd.choice(['YOU ROBBER!😡', 'YOU JEALOUS!😡'])
            return 'D'
        if rnd.random() <= 0.1:
            internal_memory[408] = max(1, internal_memory[408])
            return 'D'
        return history[-1][opp_index]
    elif strategy == 'ZD-MD / Zero Determinant Middle':
        if not len(history):
            return 'C'
        phi = 0.5
        l = (self_payoff_matrix['DD'] + self_payoff_matrix['CC']) / 2
        s_min = -min((self_payoff_matrix['DC'] - l) / (l - self_payoff_matrix['CD']), (l - self_payoff_matrix['CD']) / (self_payoff_matrix['DC'] - l))
        s = (s_min + 1) / 2
        p1 = 1 - phi * (1 - s) * (self_payoff_matrix['CC'] - l)
        p2 = 1 - phi * (s * (l - self_payoff_matrix['CD']) + (self_payoff_matrix['DC'] - l))
        p3 = phi * ((l - self_payoff_matrix['CD']) + s * (self_payoff_matrix['DC'] - l))
        p4 = phi * (1 - s) * (l - self_payoff_matrix['DD'])
        four_vector = [p1, p2, p3, p4]
        return 'C' if rnd.random() <= four_vector[((2 * (history[-1][self_index] == 'D')) + (history[-1][opp_index] == 'D'))] else 'D'
    elif strategy == 'Hill Climb Attariq Version':
        if not len(history):
            return 'C'
        d = 1 / 10
        if history[-1][opp_index] == 'C':
            if(internal_memory[422] + d >= 1):
                internal_memory[422] = (internal_memory[422] + 1) / 2
            else:
                internal_memory[422] += d
        else:
            if(internal_memory[422] - d <= 0):
                internal_memory[422] = internal_memory[422] / 2
            else:
                internal_memory[422] -= d
        return rnd_prob(internal_memory[422])
    elif strategy == 'CC-L / Cosmos-Chaos Learner':
        payoff = {
            ('C', 'C'): self_payoff_matrix['CC'],  
            ('C', 'D'): self_payoff_matrix['CD'],  
            ('D', 'C'): self_payoff_matrix['DC'],  
            ('D', 'D'): self_payoff_matrix['DD']   
        }
        weight = internal_memory[423][0]
        last_turn = internal_memory[423][1]
        if len(history) > 0:
            cond = (history[-1][self_index], history[-1][opp_index])
            if last_turn == 'COSMOS':
                weight[0] += (payoff[cond] - self_avg_payoff_matrix)
            else:
                weight[1] += (payoff[cond] - self_avg_payoff_matrix)
        cosmos = call_strategy('Risky Q-Learner', history, opp_index, opp_name, internal_memory[423][2], None, None, None, score, None)
        p = (mth.exp(weight[0]) / sum(mth.exp(weight[i]) for i in range(len(weight))))
        if rnd.random() <= p:
            internal_memory[423][1] = 'COSMOS'
            return cosmos
        internal_memory[423][1] = 'CHAOS'
        return rnd_prob(0.5)
    elif strategy == 'YOUR CATEGORY':
        if categories[1] == 'NASTY':
            return 'D'
        elif categories[1] == 'NETRAL':
            return rnd_prob(0.5)
        return 'C'
    elif strategy == 'Zero Input Euler':
        def S_function(x):
            return ((x / (abs(x) + 1)) + 1) / 2
        return rnd_prob(S_function(constant('Euler')))
    elif strategy == 'Zero Input Pi':
        def S_function(x):
            return ((x / (abs(x) + 1)) + 1) / 2
        return rnd_prob(S_function(constant('Pi')))
    elif strategy == 'Zero Input Phi':
        def S_function(x):
            return ((x / (abs(x) + 1)) + 1) / 2
        return rnd_prob(S_function(constant('Phi')))
    elif strategy == 'Zero Input Champernowne':
        def S_function(x):
            return ((x / (abs(x) + 1)) + 1) / 2
        return rnd_prob(S_function(constant('Champernowne')))
    elif strategy == 'Zero Input Tau':
        def S_function(x):
            return ((x / (abs(x) + 1)) + 1) / 2
        return rnd_prob(S_function(constant('Tau')))
    elif strategy == 'Zero Input Apery':
        def S_function(x):
            return ((x / (abs(x) + 1)) + 1) / 2
        return rnd_prob(S_function(constant('Apery')))
    elif strategy == 'Zero Input Euler-Mascheroni':
        def S_function(x):
            return ((x / (abs(x) + 1)) + 1) / 2
        return rnd_prob(S_function(constant('Euler-Mascheroni')))
    elif strategy == 'Zero Input Pythagoras':
        def S_function(x):
            return ((x / (abs(x) + 1)) + 1) / 2
        return rnd_prob(S_function(constant('Pythagoras')))
    elif strategy == 'Champernowne':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'C':
            internal_memory[430][0] += 1
        else:
            internal_memory[430][1] += 1
        if history[-1][self_index] == 'C':
            internal_memory[430][2] += 1
        else:
            internal_memory[430][3] += 1
        if internal_memory[430][1] == 0:
            return 'D'
        cooperations = internal_memory[430][0] + internal_memory[430][2]
        defections = internal_memory[430][1] + internal_memory[430][3]
        if cooperations / defections > internal_memory[430][4]:
            return 'D'
        return 'C'
    elif strategy == 'Tau':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'C':
            internal_memory[431][0] += 1
        else:
            internal_memory[431][1] += 1
        if history[-1][self_index] == 'C':
            internal_memory[431][2] += 1
        else:
            internal_memory[431][3] += 1
        if internal_memory[431][1] == 0:
            return 'D'
        cooperations = internal_memory[431][0] + internal_memory[431][2]
        defections = internal_memory[431][1] + internal_memory[431][3]
        if cooperations / defections > internal_memory[431][4]:
            return 'D'
        return 'C'
    elif strategy == 'Apery':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'C':
            internal_memory[432][0] += 1
        else:
            internal_memory[432][1] += 1
        if history[-1][self_index] == 'C':
            internal_memory[432][2] += 1
        else:
            internal_memory[432][3] += 1
        if internal_memory[432][1] == 0:
            return 'D'
        cooperations = internal_memory[432][0] + internal_memory[432][2]
        defections = internal_memory[432][1] + internal_memory[432][3]
        if cooperations / defections > internal_memory[432][4]:
            return 'D'
        return 'C'
    elif strategy == 'Euler-Mascheroni':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'C':
            internal_memory[433][0] += 1
        else:
            internal_memory[433][1] += 1
        if history[-1][self_index] == 'C':
            internal_memory[433][2] += 1
        else:
            internal_memory[433][3] += 1
        if internal_memory[433][1] == 0:
            return 'D'
        cooperations = internal_memory[433][0] + internal_memory[433][2]
        defections = internal_memory[433][1] + internal_memory[433][3]
        if cooperations / defections > internal_memory[433][4]:
            return 'D'
        return 'C'
    elif strategy == 'Pythagoras':
        if not len(history):
            return 'C'
        if history[-1][opp_index] == 'C':
            internal_memory[434][0] += 1
        else:
            internal_memory[434][1] += 1
        if history[-1][self_index] == 'C':
            internal_memory[434][2] += 1
        else:
            internal_memory[434][3] += 1
        if internal_memory[434][1] == 0:
            return 'D'
        cooperations = internal_memory[434][0] + internal_memory[434][2]
        defections = internal_memory[434][1] + internal_memory[434][3]
        if cooperations / defections > internal_memory[434][4]:
            return 'D'
        return 'C'
    elif strategy == 'Attariq\'s Cheating Matching Pennies Strategy':
        if (self_payoff_matrix['CC'] + self_payoff_matrix['DD']) > (self_payoff_matrix['CD'] + self_payoff_matrix['DC']):
            return clone_opp(rnd_prob(0.5), strategy, history, opp_index, opp_name, internal_memory, immortal_memory, chats, next_chat, score, population)
        return flip(clone_opp(rnd_prob(0.5), strategy, history, opp_index, opp_name, internal_memory, immortal_memory, chats, next_chat, score, population))
    elif strategy == 'Adaptable Game type Strategy':
        if game_type == 'Prisoner\'s Dilemma':
            if not len(history):
                return 'C'
            return 'D' if (history[-1][opp_index] == 'D') and (((history[-2][opp_index] == 'D') if len(history) > 1 else False) or (rnd.random() <= 0.5)) else 'C'
        elif game_type == 'Game Of Chicken / Hawk-Dove':
            if not len(history):
                return 'C'
            return history[-1][self_index] if self_payoff_matrix[history[-1][self_index] + history[-1][opp_index]] >= self_avg_payoff_matrix else flip(history[-1][self_index])
        elif game_type in ('Stag Hunt', 'Battle Of Sexes'):
            return clone_opp(rnd_prob(0.5), strategy, history, opp_index, opp_name, internal_memory, immortal_memory, chats, next_chat, score, population)
        elif game_type == 'Matching Pennies':
            if (self_payoff_matrix['CC'] + self_payoff_matrix['DD']) > (self_payoff_matrix['CD'] + self_payoff_matrix['DC']):
                return clone_opp(rnd_prob(0.5), strategy, history, opp_index, opp_name, internal_memory, immortal_memory, chats, next_chat, score, population)
            return flip(clone_opp(rnd_prob(0.5), strategy, history, opp_index, opp_name, internal_memory, immortal_memory, chats, next_chat, score, population))
        return rnd_prob(0.5)
    elif strategy == 'Modified Primer\'s Strategy':
        def C_function(x):
            return (self_payoff_matrix['CC'] * x) + (self_payoff_matrix['CD'] * (1 - x))
        def D_function(x):
            return (self_payoff_matrix['DC'] * x) + (self_payoff_matrix['DD'] * (1 - x))
        if len(history) < 2:
            return 'C' if not len(history) else history[-1][opp_index]
        p_c = population.count('NICE') / len(population)
        A1 = frequency('C', 'C', history[-20:], opp_index)
        A2 = frequency('D', 'C', history[-20:], opp_index)
        E_c = C_function(p_c) * C_function(A1 if A1 != None else 0.5)
        E_d = D_function(p_c) * D_function(A2 if A2 != None else 0.5)
        return 'C' if E_c >= E_d else 'D'
    elif strategy == 'Emotional Tit For Tat':
        brain = internal_memory[443]
        if brain["F"] != 'UNFORGIVABLE' and brain["F"] > 0:
            if history[-1][opp_index] == 'D':
                brain["F"] -= 1
                if brain["F"] == 0:
                    brain["F"] = 'UNFORGIVABLE'
                return 'C'
            else:
                brain["F"] = 0
        if len(history) > 0:
            if self_payoff_matrix[history[-1][self_index] + history[-1][opp_index]] < self_avg_payoff_matrix:
                min_payoff_matrix = min(self_payoff_matrix['CC'], self_payoff_matrix['CD'], self_payoff_matrix['DC'], self_payoff_matrix['DD'])
                max_payoff_matrix = max(self_payoff_matrix['CC'], self_payoff_matrix['CD'], self_payoff_matrix['DC'], self_payoff_matrix['DD'])
                brain["CORT"] += ((max_payoff_matrix - min_payoff_matrix) - (self_payoff_matrix[history[-1][self_index] + history[-1][opp_index]] - min_payoff_matrix)) / tournament_avg_last_round
            else:
                min_payoff_matrix = min(self_payoff_matrix['CC'], self_payoff_matrix['CD'], self_payoff_matrix['DC'], self_payoff_matrix['DD'])
                brain["DOPA"] += (self_payoff_matrix[history[-1][self_index] + history[-1][opp_index]] - min_payoff_matrix) / tournament_avg_last_round
        emotion = (brain["DOPA"] - brain["CORT"]) / (1 + mth.exp((brain["W"] * (len(history) - 1)) + brain["B"]))
        if brain["CORT"] > brain["DOPA"]:
            brain["W"] -= (0.1 * (1 + brain["CORT"]))
            brain["B"] -= (0.1 * (1 + brain["CORT"]))
        else:
            brain["W"] += (0.1 * (1 + brain["DOPA"]))
            brain["B"] += (0.1 * (1 + brain["DOPA"]))
        brain["W"] = min(2.0, max(0.01, round(brain["W"], 2)))
        brain["B"] = round(brain["B"], 2)
        if ((emotion < 0) and (len(history) > 0)) and ((brain["F"] != 'UNFORGIVABLE') or (history[-1][opp_index] == 'C')):
            brain["F"] = 10
            return 'C'
        return history[-1][opp_index] if len(history) > 0 else 'C'
    else:
        return None

def run_match(args):
    i, j, with_noise_flag, noise_prob_local = args
    global memory, game_score, strat_memory
    global strategy_chat, strategy_next_chat, game_round
    memory = []
    game_score = [0, 0]
    strat_memory = [reset_internal_memory(0), reset_internal_memory(1)]
    strategy_chat = ['', '']
    strategy_next_chat = [[''], ['']]
    game_round = 0
    local_round = tournament_avg_round + rnd.randint(-tournament_round_randomness, tournament_round_randomness)
    while game_round < local_round:
        strategy_next_chat = [[''], ['']]
        c1 = call_strategy(strategies[i], memory, 1, strategies[j], strat_memory[0], strat_immortal_memory[0], strategy_chat, strategy_next_chat[0], game_score, population_type)
        c2 = call_strategy(strategies[j], memory, 0, strategies[i], strat_memory[1], strat_immortal_memory[1], strategy_chat, strategy_next_chat[1], game_score, population_type)
        if with_noise_flag:
            if rnd.random() <= noise_prob_local:
                c1 = rnd_prob(0.5)
            if rnd.random() <= noise_prob_local:
                c2 = rnd_prob(0.5)
        if (c1 not in ('C', 'D', '[OUT]')) or (c2 not in ('C', 'D', '[OUT]')):
            ERROR = []
            if c1 not in ('C', 'D', '[OUT]'):
                ERROR.append(strategies[i])
            if c2 not in ('C', 'D', '[OUT]'):
                ERROR.append(strategies[j])
            raise NameError(f'ERROR {ERROR}')
        if '[OUT]' in (c1, c2):
            break
        memory.append((c1, c2))
        if c1 == c2:
            update_score(game_payoff_matrix[0]['CC'], game_payoff_matrix[1]['CC']) if c1 == 'C' else update_score(game_payoff_matrix[0]['DD'], game_payoff_matrix[1]['DD'])
        else:
            update_score(game_payoff_matrix[0]['DC'], game_payoff_matrix[1]['CD']) if c1 == 'D' else update_score(game_payoff_matrix[0]['CD'], game_payoff_matrix[1]['DC'])
        strategy_chat[0] = strategy_next_chat[0][0]
        strategy_chat[1] = strategy_next_chat[1][0]
        game_round += 1
    return (i, j, game_score[0], game_score[1])

def run_clasification(args):
    i, j, with_noise_flag, noise_prob_local = args
    global memory, game_score, strat_memory, strat_immortal_memory
    global strategy_coop_total, strategy_chat, strategy_next_chat, game_round
    memory = []
    game_score = [0, 0]
    coop_total = [0, 0]
    strat_memory = [reset_internal_memory(0), reset_internal_memory(1)]
    if rnd.random() <= 0.5:
        strat_immortal_memory = [reset_immortal_memory(0), reset_immortal_memory(1)]
    strategy_chat = ['', '']
    strategy_next_chat = [[''], ['']]
    game_round = 0
    local_round = tournament_avg_round + rnd.randint(-tournament_round_randomness, tournament_round_randomness)
    while game_round < local_round:
        strategy_next_chat = [[''], ['']]
        c1 = call_strategy(strategies[i], memory, 1, strategies[j], strat_memory[0], strat_immortal_memory[0], strategy_chat, strategy_next_chat[0], game_score, population_type)
        c2 = call_strategy(strategies[j], memory, 0, strategies[i], strat_memory[1], strat_immortal_memory[1], strategy_chat, strategy_next_chat[1], game_score, population_type)
        if with_noise_flag:
            if rnd.random() <= noise_prob_local:
                c1 = rnd_prob(0.5)
            if rnd.random() <= noise_prob_local:
                c2 = rnd_prob(0.5)
        if (c1 not in ('C', 'D', '[OUT]')) or (c2 not in ('C', 'D', '[OUT]')):
            ERROR = []
            if c1 not in ('C', 'D', '[OUT]'):
                ERROR.append(strategies[i])
            if c2 not in ('C', 'D', '[OUT]'):
                ERROR.append(strategies[j])
            raise NameError(f'ERROR {ERROR}')
        if '[OUT]' in (c1, c2):
            break
        coop_total[0] += (c1 == 'C')
        coop_total[1] += (c2 == 'C')
        memory.append((c1, c2))
        if c1 == c2:
            update_score(game_payoff_matrix[0]['CC'], game_payoff_matrix[1]['CC']) if c1 == 'C' else update_score(game_payoff_matrix[0]['DD'], game_payoff_matrix[1]['DD'])
        else:
            update_score(game_payoff_matrix[0]['DC'], game_payoff_matrix[1]['CD']) if c1 == 'D' else update_score(game_payoff_matrix[0]['CD'], game_payoff_matrix[1]['DC'])
        strategy_chat[0] = strategy_next_chat[0][0]
        strategy_chat[1] = strategy_next_chat[1][0]
        game_round += 1
    return (i, j, coop_total[0], game_round)

strategies = ['2 Tits For 2 Tats Eckhart Arnold Version', '2 Tits For Tat', '3 Seconds Mirror', '3-Tree Of Prediction', '4-Duality Optimizer 1', 'A-A&C / A-Actor&Critic', 'ALLC OR ALLD', 'AMERICA! / Freedoom / USA', 'AON2', 'AQUA', 'AZD / Adaptive Zero Determinant', 'Absurdistan 1', 'Acquaintance & Trust', 'Adaptable Game type Strategy', 'Adaptive', 'Adaptive Cooperator', 'Adaptive Hhhpraise Version', 'Adaptive Pavlov 2006', 'Adaptive Pavlov 2011', 'Adaptive Tit For Tat', 'Adaptive Tit For Tat Attariq Version', 'Adaptor Brief', 'Adaptor Long', 'Aggravater', 'Alan', 'Alcohol / Ethanol', 'Alexei / End Game Cheater', 'Alternating Strategy', 'Alternator / Second By Duisman', 'Alternator Hunter', 'Always Cooperate / Cooperator / Dove / Saint', 'Always Defect / Defector / Hawk / Shark', 'Always Out / Outtor', 'Amazon / Piraha', 'Analogy', 'Anatol', 'Angry Tit For Tat', 'Anti Bourgeois', 'Anti Cycler', 'Anti Noise Tit For Tat', 'Anti Tit For Tat / Psycho', 'Apery', 'Appeaser', 'Arrogant Q-Learner', "Attariq's Cheating Matching Pennies Strategy", 'Average Copier', 'BBE / Burn Both Ends', 'Back Stabber', 'Backrooms', 'Backward Induction', 'Bahlil', 'Bandit', 'Bandit UCB', 'Bayes Attariq Version', 'Bayes Foomii Version', 'Bayesian Forgiver', 'Benjo', 'Beta Tester', 'Better & Better', 'Bluff', 'Bluff Master', 'Boltzmann Brain', 'Bookend Cooperator', 'Bourgeois', 'Boxer', 'Bros Mind', 'Bully / Reverse Tit For Tat', 'Burn The Bridge Strategy', 'Bush Mosteller', 'Buzzer', 'CC-L / Cosmos-Chaos Learner', 'CS / Collective Strategy', 'Caerbannog', 'Calamity', 'Calculator', 'Calculator The Nerd Of AI Version', 'Capitalist', 'Capri', 'Cautious Q-Learner', 'Caveman', 'Cerebrum', 'Champernowne', 'Charity', 'Chatter', 'Chimera', 'Clement Coldridge', 'Coin', 'Collatz', 'Communalist', 'Continuous Q-Learner Avg', 'Continuous Q-Learner Max', 'Contrite Grudger', 'Contrite Tit For Tat', 'Control 1', 'Control 10', 'Control 11', 'Control 2', 'Control 3', 'Control 4', 'Control 5', 'Control 6', 'Control 7', 'Control 8', 'Control 9', 'Cooperate ISO', 'Cooperate Once', 'Cooperator Hunter', 'Crabby', 'Crabby Attariq Version', 'Cruelity', 'Cult Bishop A', 'Cult Bishop B', 'Cult Citizen A', 'Cult Citizen B', 'Cult Citizen C', 'Cult Leader', 'Curiosity', 'Cycle Hunter', 'Cycler CCCCCD', 'Cycler CCCD', 'Cycler CCCDCD', 'Cycler CCD', 'Cycler DC', 'Cycler DCCCC', 'Cycler DDC', 'DBS / Derived Belief Strategy', 'DDoS', 'Darwin', 'Deadlock Breaker', 'Decay Given', 'Defect Once', 'Defector Hunter', 'Delayed AON1', 'Desire', 'Desprate', 'Detective', 'Devil Staircase', 'Diplomat', 'Dont Bite The Hand That Feeds You', 'Double Crosser', 'Double Q-Learner', 'Double Resurrection', 'Doubler', 'Downing Benjjo Version', 'Downing Eckhart Arnold Version', 'Dumpty 10', 'Dyna Q-Learner', 'Dynamic 2 Tits For Tat', 'Dynocat', 'EGOist', 'Easy Go / Fool Me Forever / Reverse Grudger', 'Echo Adaptive', 'Echo Fisher', 'Egoist A-Learner', 'Eliza / Therapist', 'Emotional Tit For Tat', 'Entropy Sentinel', 'Envy', 'Epsilon-Greedy', 'Eric', 'Eugine Nier', 'Euler', 'Euler-Mascheroni', 'Eventual Cycle Hunter', 'Evil Alliance', 'Evolvable Cycler Attariq Version', 'Evolved FSM 16', 'Evolved FSM 16 Noise 05', 'Evolved FSM 4', 'Evolved FSM 6', 'Evolved HMM 5', 'Evolved Looker Up 1_1_1', 'Evolved Looker Up 2_2_2', 'Exploratory Lenient Grim 2', 'Exploratory Tit For 3 Tats', 'FAWS', 'False Cooperator', 'Falsity', 'Female / Woman', 'Fibonacci', 'Fictitious Play', 'Finger Print', 'Firm But Fair / Firm For Tat', 'First By Davis', 'First By Downing', 'First By Downing, But Get Revised By Axelrod', 'First By Feld', 'First By Graaskamp / Second By Mauk', 'First By Grofman', 'First By Grudger / Grim Trigger', 'First By Joss / Naive Prober', 'First By Named Withheld', 'First By Nydegger', 'First By Shubik', 'First By Stein & Rapoport', 'First By Tideman & Chieruzzi', 'First By Tit For Tat', 'First By Tullock', 'Fixed Pattern', 'Focal Point Strategy / Schelling Point Strategy', 'Fool Me Once', 'Forest-RND', 'Forest-XG', 'Forgetable Gradual', 'Forgetful Fool Me Once', 'Forgetful Grudger', 'Forgetful Majority Bot', 'Forgiver', 'Forgiver Benjamin M. Zagorsky & Johannes G. Reiter & Krishnendu Chatterjee & Martin A. Nowak Version / S14', 'Forgiving', 'Forgiving Fate', 'Forgiving Grudger', 'Forgiving Tit For Tat', 'Fortress 3', 'Fortress 4', 'Frankl', 'Free Rider', 'Frequency Analyzer', 'Freud', 'Friendly Human', 'GT-E / Game Theory Explorer', 'Gambler Attariq Version', 'Gaslighter', 'Geller', 'Geller Cooperator', 'Geller Defector', 'Gen 1_64_200_30_0.3_A', 'Gen 2_64_200_30_0.3_A', 'Gen 3_64_200_30_0.3_A', 'Gen 4_64_200_30_0.3_A', 'Gen 5_64_200_30_0.3_A', 'Generic Downing', 'Generic Downing Friendly', 'Generous Tit For Tat', 'Generous Tit For Tat Axelrod Project Contributor Team Version', 'Generous Tit For Tat Benjjo Version', 'Generous Tit For Tat Eckhart Arnold Version 1', 'Generous Tit For Tat Eckhart Arnold Version 2', 'Generous Tit For Tat Less Wrong Version', 'Genetic Algo 1', 'Genetic Algo 2', 'George', 'Go By Majority / Second By Grisell / Soft Go By Majority', 'Go By Minority', 'Gold Digger', 'Golden Mean', 'Good', 'GossipAgent', 'Graaskamp Benjjo Version', 'Gradual', 'Gradual Cristal Version', 'Gradual Killer', 'Grateful / S9', 'Greedy', 'Grudger Alternator', 'Grumpy', 'HMM Player', 'Half Sin Square', 'Handshake', 'Hard Go By Majority', 'Hard Prober', 'Hard Tit For 2 Tats', 'Hard Tit For Tat', 'Hesitant Q-Learner', 'Hill Climb', 'Hill Climb Attariq Version', 'Hopeless', 'Hot Coffee', 'Humpty 10', 'Hungry', 'ISO', 'Inequality / SOVIET UNION! / USSR', 'Inequity Averse Q-Learner', 'Inverse', 'Inverse Punisher', 'Is There Something New?', 'Isolated Clan', 'Isolated Tit For Tat', 'Japan Empire Emperor', 'Japan Empire Soldier 1', 'Japan Empire Soldier 10', 'Japan Empire Soldier 2', 'Japan Empire Soldier 3', 'Japan Empire Soldier 4', 'Japan Empire Soldier 5', 'Japan Empire Soldier 6', 'Japan Empire Soldier 7', 'Japan Empire Soldier 8', 'Japan Empire Soldier 9', 'Javelin', 'Jem', 'Jester', 'Jesus Petry', 'Josuah', 'Jung', 'Karen', 'Keyman', 'Killer', 'King Of 48 Laws Of Power', 'Knowledgeable Worse & Worse', 'Konflikt Blue', 'Konflikt Green', 'L-BODOR / Learning Based On Derivative Of Reward', 'LCG', 'Laran', 'Laws', 'Learner', 'Lenient Grim 2', 'Lenient Grim 3', 'Less Grim', 'Level Punisher', 'Liar Person', 'Lighty-Darky', 'Limited Retaliate', 'Limited Retaliate 2', 'Limited Retaliate 3', 'Line', 'Little Grudger', 'Local Copycat', 'Long Horse', 'Long Term Strategist', 'Longterm Tit For Tat', 'Look Ahead / Look Up', 'Loyal', 'MARS / Mimicry And Relative Similarity', 'MCMC / Markov Chain Monte Carlo', 'MEM2', 'MENACE', 'MSAgent', 'Mafia A', 'Mafia B', 'Mafia C', 'Mafia D', 'Majapahit', 'Male / Man', 'Malthrin', 'Man In The Middle', 'Manipulator', 'Manipulity', 'Markov', 'Matcher', 'Math Constant Hunter', 'Memory 1', 'Memory Decay', 'Mensa', 'Meta Hunter', 'Meta Hunter Aggressive', 'Meta Majority', 'Meta Minority', 'Meta Mixer', 'Meta Winner', 'Meta Winner Ensemble', 'Michaelos', 'Minimax Strategy', 'Mirror', 'Mixed Strategy', 'Modal Defector', 'Modal Tit For Tat', "Modified Primer's Strategy", 'Momentum', 'Monkey See, Monkey Do', 'Mosquito', 'MrX', 'Mystical Logic', 'N Pavlov', 'N Tits For M Tats', 'Naive Q-Learner', 'Nash', 'Nasty Tit For Tat', 'Negation', 'Neil A', 'Newton', 'Nice Meta Winner', 'Nice Meta Winner Ensemble', 'Nydegger Benjjo Version', 'Observant', 'Odd-Even Go By Majority', 'Odysseus / Ὀδυσσεύς', 'Omega Tit For Tat', 'Once Bitten', 'Opportunist', 'Opposite Grudger', 'Out For Tat', 'PIN', 'PSO Gambler 1_1_1', 'PSO Gambler 2_2_2', 'PSO Gambler 2_2_2 Noise 05', 'PSO Gambler Mem 1', 'Paradoxic Grim / S21', 'Parameterized Tit For Tat', 'Paranoid', 'Passive Tit For Tat', 'Patient Cooperator', 'Pattern Reader', 'Patterned Adaptive Fortress', 'Patterned Adaptive Fortress 2', 'Pavlov / Win-Stay, Lose-Shift', 'Pavlov D / Suspicious Pavlov', 'Pavlov Tester', 'Pavolovo', 'Pawlow', 'Payoff-Dominance Strategy', 'Perceptron', 'Phi', 'Pi', 'Prabowo', 'Praedator', 'Predator', "Primer's Strategy", 'Probable Coop', 'Probe & Exploit', 'Probe & Punish', 'Prober', 'Prober 2', 'Prober 3', 'Prober 4', 'Prober Abdullah-arif-swe Version', 'Procastinater', 'Psycho Attariq Version', 'Pun 1', 'Punisher', 'Pythagoras', 'Q-Learning A-Learner', 'REINFORCE', 'RLAgent', 'RLGossipAgent', 'RNN & RTRL', 'Racister', 'Rack Block Shooter', 'Raider', 'Random', 'Random Hunter', 'Random Tit For Tat', "Random's Exploiter 1", "Random's Exploiter 2", "Random's Exploiter 3", "Random's Exploiter 4", 'Reactive Player', 'Rememberable Grudger', 'Remorseful Prober', 'Resurrection', 'Retaliate', 'Retaliate 2', 'Retaliate 3', 'Rice Field', 'Ripoff', 'Risk-Dominance Strategy', 'Risky Q-Learner', 'Rogers & Maslow', 'Rover', 'Rwallace', 'S11', 'S12 / Suspicious Forgiver', 'S18', 'S19', 'S2', 'S23', 'S24', 'S6', 'S7', 'SURPRISE ATTACK!', 'Sample / Second By Tit For 2 Tats', 'Sarsa Q-Learner', 'Schwoba Sen Gnitz', 'Scorched Earth', 'Score Equalizer', 'Second By Almy', 'Second By Ambuehl & Hickey', 'Second By Anderson', 'Second By Appold', 'Second By Batell', 'Second By Black', 'Second By Borufsen', 'Second By Cave', 'Second By Champion', 'Second By Colbert', 'Second By Dawes & Batell', 'Second By Eatherley', 'Second By Falk & Langsted', 'Second By Feld', 'Second By Friedland', 'Second By Friedland, But Get Revised By Axelrod', "Second By GRASR's Author", 'Second By George Hufford', 'Second By Getzler', 'Second By Gladstein / Tester', 'Second By Graaskamp & Katzen', 'Second By Grofman', 'Second By Hall', 'Second By Harrington', 'Second By Hollander', 'Second By Hotz', 'Second By Jones', 'Second By Joss', 'Second By Kluepfel', 'Second By Lefevre', 'Second By Leyland', 'Second By Leyvraz', 'Second By Mcgurrin', 'Second By Mikkelson', 'Second By Newman', 'Second By Nussbacher', 'Second By Nydegger', 'Second By Pebly', 'Second By Pinkley', 'Second By Quayle', 'Second By Rabbie', 'Second By Richard Hufford', 'Second By Robert Adams', 'Second By Robertson', 'Second By Rowsam', 'Second By Schurmann', 'Second By Smith', 'Second By Smoody', 'Second By Snodgrass', 'Second By Tideman & Chieruzzi', 'Second By Tranquilizer', 'Second By Weiderman', 'Second By Weiner', 'Second By White', 'Second By Wiliam Adams', 'Second By Yamachi', 'Second By Zimmerman', 'Second Chance', 'Second Chance Attariq Version', 'Self Steem', 'Self-Love A-Learner', 'Short Mem', 'Signaling Cheater', 'Simple ANN', 'Simple Exploiter', 'Simple Identity ChecK', 'Simpleton', 'Skinner', 'Slanger', 'Slow Tit For Tat', 'Smart', 'Smart Cat', 'Smart Generous Tit For Tat', 'Smart Go By Majority', 'Smart Snake', 'Smart-Clever', 'Smarter Tat For Tit', 'Smooth Grudger', 'Smooth Tit For Tat', 'Sneaky Tit For Tat', 'Social Average', 'Social Engineering', 'Socio', 'Soft Grim', 'Soft Grudger', 'Solution B1', 'Solution B5', 'Sort', 'Spectrum Attacker', 'Spiteful CC', 'Spiteful Tit For Tat', 'Split Brain Syndrome', 'Springter', 'Stabilize', 'Stalker', 'Star S', 'Star SL', 'Star SN', 'Star Slave 1', 'Star Slave 10', 'Star Slave 2', 'Star Slave 3', 'Star Slave 4', 'Star Slave 5', 'Star Slave 6', 'Star Slave 7', 'Star Slave 8', 'Star Slave 9', 'Statistical Strategy', 'Stefan Pennartz', 'Stochastic Cooperator', 'Stochastic WSLS', 'Stoic Mirror', 'Stoicalist', 'Strategic Commitment', 'Strategist', 'Stupidiot', 'Sugar Strategy / Treasure Hunt', 'Sumobox Fighter', 'Sun Tzu Bot', 'Suspicious Alternator', 'Suspicious Tit For Tat', 'Sweet Heart', 'Syifa', 'TF 1', 'TF 2', 'TF 3', 'TF2T & 2TFT', 'TOM Level 1 / Theory Of Mind Level 1', 'TOM Level 2 / Theory Of Mind Level 2', 'TOM Level 3 / Theory Of Mind Level 3', 'TPMS / Tree Predictive Matrix Strategy', 'Tages', 'Tau', 'Teacher Pee While Standing, Student Pee While Running', 'Tester Eckhart Arnold Version', 'The Calculated Mirror With A Fuse', 'The Cave Breaker', 'The Nasty A', 'Theseus', 'Three Times Out', 'Threshold Punisher', 'Thresholded Trust Bot', 'Thue Morse', 'Thue Morse Inverse', 'Thumper', 'Tiny Brain', 'Tip For Tap', 'Tit For 3 Tats', 'Tit For Increasing Tat', 'Tit For Tat But Not Too Forgiving', 'Tit For Tat Killer', 'Toxic Mirror', 'Trader', 'Tranquilizer Eckhart Arnold Version', 'Tri-Brain', 'Tricky Cooperator', 'Tricky Defector', 'Tricky Level Punisher', 'Trust Ledger', 'Trust Score Sigmoid', 'Trusty', 'Turan', 'Tweedledee', 'Tweedledum', 'Tweetypie', 'Unpredictable Tit For Tat', 'Uranium', 'Usually Cooperates', 'Usually Defects', 'Vandal', 'Vengeful', 'Vengeful Cheater', 'Verity', 'Very Bad', 'WHAgent', 'Whale', 'William', 'Willing', 'Win-Shift, Lose-Stay', 'Winner 12', 'Winner 21', 'Worse & Worse / Worse & Worse 1', 'Worse & Worse 2', 'Worse & Worse 3', 'YOUR CATEGORY', 'Your Choice Or My Choice', 'ZD-2026 / Zero Determinant 2026', 'ZD-G / Zero Determinant Generous', 'ZD-GTFT / Zero Determinant Generous Tit For Tat', 'ZD-M2 / Zero Determinant Memory 2', 'ZD-MD / Zero Determinant Middle', 'ZD-MS / Zero Determinant Mischief', 'ZD-PW2026 / Zero Determinant Psychological War 2026', 'ZD-Q / Zero Determinant Equalizer', 'ZD-S / Zero Determinant Set', 'ZD-X / Zero Determinant Extortion', 'ZD-X A. J. Stewart & J. B. Plotkin Version / Zero Determinant Extortion A. J. Stewart & J. B. Plotkin Version', 'ZD-X Lars Roemheld Version / Zero Determinant Extortion Lars Roemheld Version', 'ZD-X Marc Harper Version / Zero Determinant Extortion Marc Harper Version', 'ZD-X Steven Kuhn Version / Zero Determinant Extortion Steven Kuhn Version', 'Zero Day', 'Zero Input Apery', 'Zero Input Champernowne', 'Zero Input Euler', 'Zero Input Euler-Mascheroni', 'Zero Input Phi', 'Zero Input Pi', 'Zero Input Pythagoras', 'Zero Input Tau', 'ZeroResp', 'ZeroResp V5.3']
strategies_creator = {'2 Tits For 2 Tats Eckhart Arnold Version': 'Eckhart Arnold', '2 Tits For Tat': 'John Maynard Smith', '3 Seconds Mirror': 'Meta', '3-Tree Of Prediction': 'M. A. M. Attariq', '4-Duality Optimizer 1': 'Paul Franceschi', 'A-A&C / A-Actor&Critic': 'M. A. M. Attariq & Gemini', 'ALLC OR ALLD': 'Marc Harper', 'AMERICA! / Freedoom / USA': 'M. A. M. Attariq', 'AON2': 'C. Hilbe & L. A. Martinez-Vaquero & K. Chatterjee & M. A. Nowak', 'AQUA': 'M. A. M. Attariq', 'AZD / Adaptive Zero Determinant': 'Emmanuel Estrada & Dashiell Fryer', 'Absurdistan 1': 'Christian Erlen', 'Acquaintance & Trust': 'Philip Seehausen', 'Adaptable Game type Strategy': 'M. A. M. Attariq', 'Adaptive': 'J. Li & P. Hingston & G. Kendall', 'Adaptive Cooperator': 'cjgwang & Vince Knight', 'Adaptive Hhhpraise Version': 'Hhhpraise', 'Adaptive Pavlov 2006': 'J. Li', 'Adaptive Pavlov 2011': 'J. Li & P. Hingston & G. Kendall', 'Adaptive Tit For Tat': 'E. Tzafestas', 'Adaptive Tit For Tat Attariq Version': 'M. A. M. Attariq', 'Adaptor Brief': 'Christoph Hauert & Olaf Stenull', 'Adaptor Long': 'Christoph Hauert & Olaf Stenull', 'Aggravater': 'Thomas Campbell', 'Alan': 'M. A. M. Attariq', 'Alcohol / Ethanol': 'M. A. M. Attariq', 'Alexei / End Game Cheater': 'Alexei / Eckhart Arnold', 'Alternating Strategy': 'Robert Aumann', 'Alternator / Second By Duisman': 'Robert Axelrod / George Duisman', 'Alternator Hunter': 'Karol Langner', 'Always Cooperate / Cooperator / Dove / Saint': 'Martin Nowak & Karl Sigmund & Robert May / Robert Axelrod / John Maynard Smith & George R. Price / Robert Trivers & Richard Dawkins', 'Always Defect / Defector / Hawk / Shark': 'Martin Nowak & Karl Sigmund & Robert May / Robert Axelrod / John Maynard Smith & George R. Price / Robert Trivers & Richard Dawkins', 'Always Out / Outtor': 'M. A. M. Attariq', 'Amazon / Piraha': 'M. A. M. Attariq', 'Analogy': 'Robert Gladstein', 'Anatol': 'M. A. M. Attariq', 'Angry Tit For Tat': 'M. A. M. Attariq', 'Anti Bourgeois': 'John Maynard Smith & George R. Price', 'Anti Cycler': 'Marc Harper', 'Anti Noise Tit For Tat': 'M. A. M. Attariq', 'Anti Tit For Tat / Psycho': 'C. Hilbe & M. A. Nowak & A. Traulsen / D. Ashlock & E. Y. Kim & W. Ashlock', 'Apery': 'M. A. M. Attariq', 'Appeaser': 'Jochen Muller', 'Arrogant Q-Learner': 'Geraint Palmer', "Attariq's Cheating Matching Pennies Strategy": 'M. A. M. Attariq', 'Average Copier': 'Geraint Palmer', 'BBE / Burn Both Ends': 'Louis Marinoff', 'Back Stabber': 'Thomas Campbell', 'Backrooms': 'M. A. M. Attariq', 'Backward Induction': 'Reinhard Selten', 'Bahlil': 'M. A. M. Attariq', 'Bandit': 'M. A. M. Attariq & Gemini', 'Bandit UCB': 'Auer & Cesa-Bianchi & Fischer', 'Bayes Attariq Version': 'M. A. M. Attariq', 'Bayes Foomii Version': 'Foomii', 'Bayesian Forgiver': 'Matt Hodges', 'Benjo': 'benjjo', 'Beta Tester': 'M. A. M. Attariq', 'Better & Better': "LIFL(Laboratoire d'Informatique Fondamentale de Lille)", 'Bluff': 'M. A. M. Attariq & Anonymous3', 'Bluff Master': 'Perplexity', 'Boltzmann Brain': 'M. A. M. Attariq', 'Bookend Cooperator': 'Explerify.com', 'Bourgeois': 'John Maynard Smith & George R. Price', 'Boxer': 'A. R. L. R. Cunha & A. L. V. Coelho', 'Bros Mind': 'A. R. L. R. Cunha & A. L. V. Coelho', 'Bully / Reverse Tit For Tat': 'John Nachbar', 'Burn The Bridge Strategy': 'tigressine', 'Bush Mosteller': 'Luis R. Izquierdo & Segismundo S. Izquierdo', 'Buzzer': 'M. A. M. Attariq', 'CC-L / Cosmos-Chaos Learner': 'M. A. M. Attariq', 'CS / Collective Strategy': 'J. Li & G. Kendall', 'Caerbannog': 'Caerbannog', 'Calamity': 'M. A. M. Attariq', 'Calculator': "LIFL(Laboratoire d'Informatique Fondamentale de Lille)", 'Calculator The Nerd Of AI Version': 'The Nerd Of AI', 'Capitalist': 'Chat-GPT', 'Capri': 'Y. Murase And His Contributors', 'Cautious Q-Learner': 'Geraint Palmer', 'Caveman': 'M. A. M. Attariq', 'Cerebrum': 'M. A. M. Attariq & Chat-GPT & Gemini', 'Champernowne': 'M. A. M. Attariq', 'Charity': 'M. A. M. Attariq', 'Chatter': 'M. A. M. Attariq', 'Chimera': 'Joep Vullinghs', 'Clement Coldridge': 'M. A. M. Attariq', 'Coin': 'M. A. M. Attariq', 'Collatz': 'M. A. M. Attariq', 'Communalist': 'Chat-GPT', 'Continuous Q-Learner Avg': 'M. A. M. Attariq', 'Continuous Q-Learner Max': 'M. A. M. Attariq', 'Contrite Grudger': 'M. A. M. Attariq', 'Contrite Tit For Tat': 'J. Wu & Robert Axelrod', 'Control 1': '[Less Wrong]', 'Control 10': '[Less Wrong]', 'Control 11': '[Less Wrong]', 'Control 2': '[Less Wrong]', 'Control 3': '[Less Wrong]', 'Control 4': '[Less Wrong]', 'Control 5': '[Less Wrong]', 'Control 6': '[Less Wrong]', 'Control 7': '[Less Wrong]', 'Control 8': '[Less Wrong]', 'Control 9': '[Less Wrong]', 'Cooperate ISO': 'Adrian Hutter', 'Cooperate Once': 'benjjo', 'Cooperator Hunter': 'Karol Langner', 'Crabby': 'Zachary Danziger', 'Crabby Attariq Version': 'M. A. M. Attariq', 'Cruelity': 'M. A. M. Attariq', 'Cult Bishop A': 'M. A. M. Attariq', 'Cult Bishop B': 'M. A. M. Attariq', 'Cult Citizen A': 'M. A. M. Attariq', 'Cult Citizen B': 'M. A. M. Attariq', 'Cult Citizen C': 'M. A. M. Attariq', 'Cult Leader': 'M. A. M. Attariq', 'Curiosity': 'M. A. M. Attariq', 'Cycle Hunter': 'Marc Harper', 'Cycler CCCCCD': 'Marc Harper', 'Cycler CCCD': 'Marc Harper', 'Cycler CCCDCD': 'Marc Harper', 'Cycler CCD': 'Marc Harper', 'Cycler DC': 'Marc Harper', 'Cycler DCCCC': 'M. A. M. Attariq', 'Cycler DDC': 'Marc Harper', 'DBS / Derived Belief Strategy': 'T.-C. Au & D. S. Nau', 'DDoS': 'M. A. M. Attariq', 'Darwin': 'Paul Slavin', 'Deadlock Breaker': 'D. Ashlock & E. Y. Kim ', 'Decay Given': 'M. A. M. Attariq', 'Defect Once': 'benjjo', 'Defector Hunter': 'Karol Langner', 'Delayed AON1': 'C. Hilbe & L. A. Martinez-Vaquero & K. Chatterjee & M. A. Nowak', 'Desire': 'M. A. M. Attariq', 'Desprate': 'P. Van Den Berg & F. J. Weissing', 'Detective': 'Nicky Case', 'Devil Staircase': 'M. A. M. Attariq', 'Diplomat': 'Perplexity', 'Dont Bite The Hand That Feeds You': 'M. A. M. Attariq', 'Double Crosser': 'Thomas Campbell', 'Double Q-Learner': 'M. A. M. Attariq', 'Double Resurrection': 'Eckhart Arnold', 'Doubler': "LIFL(Laboratoire d'Informatique Fondamentale de Lille)", 'Downing Benjjo Version': 'benjjo', 'Downing Eckhart Arnold Version': 'Eckhart Arnold', 'Dumpty 10': 'Eckhart Arnold', 'Dyna Q-Learner': 'M. A. M. Attariq', 'Dynamic 2 Tits For Tat': 'Grant Garrett-Grossman', 'Dynocat': 'rexmhall09', 'EGOist': 'M. A. M. Attariq', 'Easy Go / Fool Me Forever / Reverse Grudger': "LIFL(Laboratoire d'Informatique Fondamentale de Lille) / Marc Harper / J. Li & P. Hingston & G. Kendall", 'Echo Adaptive': 'Copilot', 'Echo Fisher': 'M. A. M. Attariq', 'Egoist A-Learner': 'M. A. M. Attariq', 'Eliza / Therapist': 'M. A. M. Attariq', 'Emotional Tit For Tat': 'M. A. M. Attariq', 'Entropy Sentinel': 'Chat-GPT', 'Envy': 'Bernd Lahno', 'Epsilon-Greedy': 'Richard S. Sutton & Andrew G. Barto', 'Eric': 'M. A. M. Attariq & Erik & Chat-GPT', 'Eugine Nier': 'Eugine Nier', 'Euler': 'Timothy Standen', 'Euler-Mascheroni': 'M. A. M. Attariq', 'Eventual Cycle Hunter': 'Marc Harper', 'Evil Alliance': 'ArisKatsaris', 'Evolvable Cycler Attariq Version': 'M. A. M. Attariq', 'Evolved FSM 16': 'Marc Harper', 'Evolved FSM 16 Noise 05': 'Marc Harper', 'Evolved FSM 4': 'Marc Harper', 'Evolved FSM 6': 'Frederick Vincent & Dashiell Fryer', 'Evolved HMM 5': 'Marc Harper', 'Evolved Looker Up 1_1_1': 'Marc Harper', 'Evolved Looker Up 2_2_2': 'Marc Harper', 'Exploratory Lenient Grim 2': 'Yaroslav Rosokha & Julian Romero', 'Exploratory Tit For 3 Tats': 'Yaroslav Rosokha & Julian Romero', 'FAWS': 'FAWS', 'False Cooperator': 'Yaroslav Rosokha & Julian Romero', 'Falsity': 'M. A. M. Attariq', 'Female / Woman': 'M. A. M. Attariq', 'Fibonacci': 'M. A. M. Attariq', 'Fictitious Play': 'George Brown', 'Finger Print': 'Eckhart Arnold', 'Firm But Fair / Firm For Tat': 'Marcus R. Frean', 'First By Davis': 'Morton Davis', 'First By Downing': 'Leslie Downing', 'First By Downing, But Get Revised By Axelrod': 'Robert Axelrod', 'First By Feld': 'Scott Feld', 'First By Graaskamp / Second By Mauk': 'Jim Graaskamp / Fred Mauk', 'First By Grofman': 'Bernard Grofman', 'First By Grudger / Grim Trigger': 'James W. Friedman', 'First By Joss / Naive Prober': 'Johann Joss / J. Li & P. Hingston & G. Kendall', 'First By Named Withheld': 'Anonymous', 'First By Nydegger': 'Rudy Nydegger', 'First By Shubik': 'Martin Shubik', 'First By Stein & Rapoport': 'Stein & Anatol Rapoport', 'First By Tideman & Chieruzzi': 'T.Nicolaus Tideman & Paula Chieruzzi', 'First By Tit For Tat': 'Anatol Rapoport', 'First By Tullock': 'Gordon Tullock', 'Fixed Pattern': 'Eckhart Arnold', 'Focal Point Strategy / Schelling Point Strategy': 'Thomas Schelling', 'Fool Me Once': 'Marc Harper', 'Forest-RND': 'M. A. M. Attariq', 'Forest-XG': 'M. A. M. Attariq', 'Forgetable Gradual': 'M. A. M. Attariq', 'Forgetful Fool Me Once': 'Marc Harper', 'Forgetful Grudger': 'Geraint Palmer', 'Forgetful Majority Bot': 'abdullah-arif-swe', 'Forgiver': 'Thomas Campbell', 'Forgiver Benjamin M. Zagorsky & Johannes G. Reiter & Krishnendu Chatterjee & Martin A. Nowak Version / S14': 'Benjamin M. Zagorsky & Johannes G. Reiter & Krishnendu Chatterjee & Martin A. Nowak', 'Forgiving': 'Hhhpraise', 'Forgiving Fate': 'M. A. M. Attariq & Nabilah Shafirah', 'Forgiving Grudger': 'Explerify.com', 'Forgiving Tit For Tat': 'Thomas Campbell', 'Fortress 3': 'W. Ashlock & D. Ashlock', 'Fortress 4': 'W. Ashlock & D. Ashlock', 'Frankl': 'M. A. M. Attariq', 'Free Rider': 'Enquist & Leimar', 'Frequency Analyzer': 'Ian Miller', 'Freud': 'M. A. M. Attariq', 'Friendly Human': 'Alex Mainzer', 'GT-E / Game Theory Explorer': 'M. A. M. Attariq', 'Gambler Attariq Version': 'M. A. M. Attariq', 'Gaslighter': 'M. A. M. Attariq & Nabilah Shafirah', 'Geller': 'Martin Chorley', 'Geller Cooperator': 'Karol Langner', 'Geller Defector': 'Karol Langner', 'Gen 1_64_200_30_0.3_A': 'M. A. M. Attariq', 'Gen 2_64_200_30_0.3_A': 'M. A. M. Attariq', 'Gen 3_64_200_30_0.3_A': 'M. A. M. Attariq', 'Gen 4_64_200_30_0.3_A': 'M. A. M. Attariq', 'Gen 5_64_200_30_0.3_A': 'M. A. M. Attariq', 'Generic Downing': 'Sven Sommer', 'Generic Downing Friendly': 'Sven Sommer', 'Generous Tit For Tat': 'Robert Axelrod', 'Generous Tit For Tat Axelrod Project Contributor Team Version': 'Axelrod Project Contributor Team', 'Generous Tit For Tat Benjjo Version': 'benjjo', 'Generous Tit For Tat Eckhart Arnold Version 1': 'Eckhart Arnold', 'Generous Tit For Tat Eckhart Arnold Version 2': 'Eckhart Arnold', 'Generous Tit For Tat Less Wrong Version': '[Less Wrong]', 'Genetic Algo 1': 'M. A. M. Attariq', 'Genetic Algo 2': 'M. A. M. Attariq', 'George': 'M. A. M. Attariq', 'Go By Majority / Second By Grisell / Soft Go By Majority': 'Robert Axelrod / Gail Grisell / S. Mittal & K. Deb', 'Go By Minority': 'M. A. M. Attariq', 'Gold Digger': 'Foomii', 'Golden Mean': 'Dola', 'Good': 'M. A. M. Attariq', 'GossipAgent': 'remcofl', 'Graaskamp Benjjo Version': 'benjjo', 'Gradual': 'B. Beaufils & J. Delahaye & P. Mathieu', 'Gradual Cristal Version': 'CRISTAL Lab & SMAC Team', 'Gradual Killer': "LIFL(Laboratoire d'Informatique Fondamentale de Lille)", 'Grateful / S9': 'Benjamin M. Zagorsky & Johannes G. Reiter & Krishnendu Chatterjee & Martin A. Nowak', 'Greedy': 'Richard S. Sutton & Andrew G. Barto', 'Grudger Alternator': 'Geraint Palmer', 'Grumpy': 'Jason Young', 'HMM Player': 'Marc Harper', 'Half Sin Square': 'M. A. M. Attariq', 'Handshake': 'Arthur J. Robson', 'Hard Go By Majority': 'S. Mittal & K. Deb', 'Hard Prober': "LIFL(Laboratoire d'Informatique Fondamentale de Lille)", 'Hard Tit For 2 Tats': 'A. J. Stewart & J. B. Plotkin', 'Hard Tit For Tat': '[PD2017]', 'Hesitant Q-Learner': 'Geraint Palmer', 'Hill Climb': 'jodoyle29', 'Hill Climb Attariq Version': 'M. A. M. Attariq', 'Hopeless': 'P. Van Den Berg & F. J. Weissing', 'Hot Coffee': 'M. A. M. Attariq', 'Humpty 10': 'Eckhart Arnold', 'Hungry': 'M. A. M. Attariq', 'ISO': 'Adrian Hutter', 'Inequality / SOVIET UNION! / USSR': 'M. A. M. Attariq', 'Inequity Averse Q-Learner': 'M. A. M. Attariq', 'Inverse': 'Karol Langner', 'Inverse Punisher': 'Geraint Palmer', 'Is There Something New?': 'M. A. M. Attariq', 'Isolated Clan': 'M. A. M. Attariq', 'Isolated Tit For Tat': 'M. A. M. Attariq', 'Japan Empire Emperor': 'M. A. M. Attariq', 'Japan Empire Soldier 1': 'M. A. M. Attariq', 'Japan Empire Soldier 10': 'M. A. M. Attariq', 'Japan Empire Soldier 2': 'M. A. M. Attariq', 'Japan Empire Soldier 3': 'M. A. M. Attariq', 'Japan Empire Soldier 4': 'M. A. M. Attariq', 'Japan Empire Soldier 5': 'M. A. M. Attariq', 'Japan Empire Soldier 6': 'M. A. M. Attariq', 'Japan Empire Soldier 7': 'M. A. M. Attariq', 'Japan Empire Soldier 8': 'M. A. M. Attariq', 'Japan Empire Soldier 9': 'M. A. M. Attariq', 'Javelin': 'M. A. M. Attariq & Angga Febriansyah', 'Jem': 'Jem', 'Jester': 'parzhitsky', 'Jesus Petry': 'Jesus Petry', 'Josuah': 'M. A. M. Attariq', 'Jung': 'M. A. M. Attariq', 'Karen': 'M. A. M. Attariq', 'Keyman': 'M. A. M. Attariq', 'Killer': 'Zachary Danziger', 'King Of 48 Laws Of Power': 'M. A. M. Attariq', 'Knowledgeable Worse & Worse': 'Adam Pohl', 'Konflikt Blue': 'M. A. M. Attariq', 'Konflikt Green': 'M. A. M. Attariq', 'L-BODOR / Learning Based On Derivative Of Reward': 'M. A. M. Attariq', 'LCG': 'M. A. M. Attariq', 'Laran': 'Elio Piccolo & Giovanni Squillero', 'Laws': 'M. A. M. Attariq', 'Learner': 'M. A. M. Attariq', 'Lenient Grim 2': 'Yaroslav Rosokha & Julian Romero', 'Lenient Grim 3': 'Yaroslav Rosokha & Julian Romero', 'Less Grim': 'Sven Sommer', 'Level Punisher': 'Eckhart Arnold', 'Liar Person': 'M. A. M. Attariq & Angga Febriansyah', 'Lighty-Darky': 'M. A. M. Attariq', 'Limited Retaliate': 'Owen Campbell', 'Limited Retaliate 2': 'Owen Campbell', 'Limited Retaliate 3': 'Owen Campbell', 'Line': 'M. A. M. Attariq', 'Little Grudger': 'M. A. M. Attariq', 'Local Copycat': 'Martin Nowak & Karl Sigmund', 'Long Horse': 'M. A. M. Attariq', 'Long Term Strategist': 'The Nerd Of AI', 'Longterm Tit For Tat': 'Adrian Hutter', 'Look Ahead / Look Up': 'Alan Gaines', 'Loyal': 'Foomii', 'MARS / Mimicry And Relative Similarity': 'T. L. B. M. Silva & A. L. V. Coelho & C. H. C. Ribeiro', 'MCMC / Markov Chain Monte Carlo': 'Nicholas Metropolis & W.K. Hastings, but implemented by Tuomas Sandholm & Robert Crites', 'MEM2': 'J. Li & G. Kendall', 'MENACE': 'M. A. M. Attariq', 'MSAgent': 'remcofl', 'Mafia A': 'M. A. M. Attariq', 'Mafia B': 'M. A. M. Attariq', 'Mafia C': 'M. A. M. Attariq', 'Mafia D': 'M. A. M. Attariq', 'Majapahit': 'M. A. M. Attariq', 'Male / Man': 'M. A. M. Attariq', 'Malthrin': 'Malthrin', 'Man In The Middle': 'M. A. M. Attariq', 'Manipulator': 'The Nerd Of AI', 'Manipulity': 'M. A. M. Attariq', 'Markov': 'M. A. M. Attariq', 'Matcher': 'M. A. M. Attariq', 'Math Constant Hunter': 'Karol Langner', 'Memory 1': 'SidZRed', 'Memory Decay': 'Axelrod Project Contributor Team', 'Mensa': 'Zachary Danziger', 'Meta Hunter': 'Karol Langner', 'Meta Hunter Aggressive': 'Marc Harper', 'Meta Majority': 'Karol Langner', 'Meta Minority': 'Karol Langner', 'Meta Mixer': 'Karol Langner', 'Meta Winner': 'Karol Langner', 'Meta Winner Ensemble': 'Marc Harper', 'Michaelos': 'Michaelos', 'Minimax Strategy': 'John von Neumann', 'Mirror': 'Foomii', 'Mixed Strategy': 'John von Neumann & Oskar Morgenstern', 'Modal Defector': 'benjjo', 'Modal Tit For Tat': 'benjjo', "Modified Primer's Strategy": 'M. A. M. Attariq', 'Momentum': 'Dong Won Moon', 'Monkey See, Monkey Do': 'M. A. M. Attariq', 'Mosquito': 'M. A. M. Attariq', 'MrX': 'Bjoern Van Den Bruck', 'Mystical Logic': 'M. A. M. Attariq', 'N Pavlov': 'SidZRed', 'N Tits For M Tats': 'Marc Harper', 'Naive Q-Learner': 'M. A. M. Attariq', 'Nash': 'M. A. M. Attariq', 'Nasty Tit For Tat': 'Axelrod Project Contributor Team', 'Negation': '[PD2017]', 'Neil A': 'M. A. M. Attariq', 'Newton': 'M. A. M. Attariq', 'Nice Meta Winner': 'Karol Langner', 'Nice Meta Winner Ensemble': 'Marc Harper', 'Nydegger Benjjo Version': 'benjjo', 'Observant': 'Zachary Danziger', 'Odd-Even Go By Majority': 'M. A. M. Attariq', 'Odysseus / Ὀδυσσεύς': 'M. A. M. Attariq', 'Omega Tit For Tat': 'W. Slany & W. Kienreich', 'Once Bitten': 'Holly Marissa', 'Opportunist': 'The Nerd Of AI', 'Opposite Grudger': 'Geraint Palmer', 'Out For Tat': 'M. A. M. Attariq', 'PIN': 'Philip Seehausen', 'PSO Gambler 1_1_1': 'Marc Harper', 'PSO Gambler 2_2_2': 'Georgios Koutsovoulos & Marc Harper', 'PSO Gambler 2_2_2 Noise 05': 'Marc Harper', 'PSO Gambler Mem 1': 'Marc Harper', 'Paradoxic Grim / S21': 'Benjamin M. Zagorsky & Johannes G. Reiter & Krishnendu Chatterjee & Martin A. Nowak', 'Parameterized Tit For Tat': 'Eckhart Arnold', 'Paranoid': 'M. A. M. Attariq', 'Passive Tit For Tat': 'M. A. M. Attariq', 'Patient Cooperator': 'Explerify.com', 'Pattern Reader': 'Explerify.com', 'Patterned Adaptive Fortress': 'M. A. M. Attariq & Claude & Gemini & Chat-GPT & Copilot', 'Patterned Adaptive Fortress 2': 'M. A. M. Attariq & Claude & Gemini & Chat-GPT & DeepSeek & Dola', 'Pavlov / Win-Stay, Lose-Shift': 'David Kraines & Vivian Kraines', 'Pavlov D / Suspicious Pavlov': 'Zhen Ji', 'Pavlov Tester': 'M. A. M. Attariq', 'Pavolovo': 'M. A. M. Attariq', 'Pawlow': 'M. A. M. Attariq', 'Payoff-Dominance Strategy': 'John Harsanyi & Reinhard Selten', 'Perceptron': 'M. A. M. Attariq', 'Phi': 'Timothy Standen', 'Pi': 'Timothy Standen', 'Prabowo': 'M. A. M. Attariq', 'Praedator': 'M. A. M. Attariq', 'Predator': 'W. Ashlock & D. Ashlock', "Primer's Strategy": 'Primer', 'Probable Coop': 'SidZRed', 'Probe & Exploit': 'Explerify.com', 'Probe & Punish': 'Eneasz', 'Prober': 'J. Li & P. Hingston & G. Kendall', 'Prober 2': "LIFL(Laboratoire d'Informatique Fondamentale de Lille)", 'Prober 3': "LIFL(Laboratoire d'Informatique Fondamentale de Lille)", 'Prober 4': "LIFL(Laboratoire d'Informatique Fondamentale de Lille)", 'Prober Abdullah-arif-swe Version': 'abdullah-arif-swe', 'Procastinater': 'M. A. M. Attariq', 'Psycho Attariq Version': 'M. A. M. Attariq', 'Pun 1': 'D. Ashlock & E. Y. Kim & N. Leahy', 'Punisher': 'Geraint Palmer', 'Pythagoras': 'M. A. M. Attariq', 'Q-Learning A-Learner': 'M. A. M. Attariq', 'REINFORCE': 'M. A. M. Attariq', 'RLAgent': 'remcofl', 'RLGossipAgent': 'remcofl', 'RNN & RTRL': 'M. A. M. Attariq', 'Racister': 'M. A. M. Attariq', 'Rack Block Shooter': 'Nic Smith', 'Raider': 'W. Ashlock & J. Tsang & D. Ashlock', 'Random': 'Robert Axelrod', 'Random Hunter': 'Karol Langner', 'Random Tit For Tat': 'Zachary M. Taylor', "Random's Exploiter 1": 'M. A. M. Attariq', "Random's Exploiter 2": 'M. A. M. Attariq', "Random's Exploiter 3": 'M. A. M. Attariq', "Random's Exploiter 4": 'M. A. M. Attariq', 'Reactive Player': 'Martin Nowak & Karl Sigmund', 'Rememberable Grudger': 'M. A. M. Attariq', 'Remorseful Prober': 'J. Li & P. Hingston & G. Kendall', 'Resurrection': 'Eckhart Arnold', 'Retaliate': 'Owen Campbell', 'Retaliate 2': 'Owen Campbell', 'Retaliate 3': 'Owen Campbell', 'Rice Field': 'Meta', 'Ripoff': 'D. Ashlock & E. Y. Kim', 'Risk-Dominance Strategy': 'John Harsanyi & Reinhard Selten', 'Risky Q-Learner': 'Geraint Palmer', 'Rogers & Maslow': 'M. A. M. Attariq', 'Rover': 'Dugatkin & Wilson', 'Rwallace': 'Rwallace', 'S11': 'Benjamin M. Zagorsky & Johannes G. Reiter & Krishnendu Chatterjee & Martin A. Nowak', 'S12 / Suspicious Forgiver': 'Benjamin M. Zagorsky & Johannes G. Reiter & Krishnendu Chatterjee & Martin A. Nowak', 'S18': 'Benjamin M. Zagorsky & Johannes G. Reiter & Krishnendu Chatterjee & Martin A. Nowak', 'S19': 'Benjamin M. Zagorsky & Johannes G. Reiter & Krishnendu Chatterjee & Martin A. Nowak', 'S2': 'Benjamin M. Zagorsky & Johannes G. Reiter & Krishnendu Chatterjee & Martin A. Nowak', 'S23': 'Benjamin M. Zagorsky & Johannes G. Reiter & Krishnendu Chatterjee & Martin A. Nowak', 'S24': 'Benjamin M. Zagorsky & Johannes G. Reiter & Krishnendu Chatterjee & Martin A. Nowak', 'S6': 'Benjamin M. Zagorsky & Johannes G. Reiter & Krishnendu Chatterjee & Martin A. Nowak', 'S7': 'Benjamin M. Zagorsky & Johannes G. Reiter & Krishnendu Chatterjee & Martin A. Nowak', 'SURPRISE ATTACK!': 'M. A. M. Attariq', 'Sample / Second By Tit For 2 Tats': 'Robert Axelrod / John Maynard Smith', 'Sarsa Q-Learner': 'M. A. M. Attariq', 'Schwoba Sen Gnitz': 'Georg Hacky Hackberger Schmidt', 'Scorched Earth': 'Explerify.com', 'Score Equalizer': 'M. A. M. Attariq', 'Second By Almy': 'Tom Almy', 'Second By Ambuehl & Hickey': 'Dennis Ambuehl & Kevin Hickey', 'Second By Anderson': 'R. D. Anderson', 'Second By Appold': 'Scott Appold', 'Second By Batell': 'Mark F. Batell', 'Second By Black': 'Paul E. Black', 'Second By Borufsen': 'Otto Borufsen', 'Second By Cave': 'Rob Cave', 'Second By Champion': 'Danny C. Champion', 'Second By Colbert': 'John Willaim Colbert', 'Second By Dawes & Batell': 'Robyn M. Dawes & Mark Batell', 'Second By Eatherley': 'Graham J. Eatherley', 'Second By Falk & Langsted': 'Robert B. Falk & James M. Langsted', 'Second By Feld': 'Scott Feld', 'Second By Friedland': 'Edward Friedland', 'Second By Friedland, But Get Revised By Axelrod': 'Robert Axelrod', "Second By GRASR's Author": 'Anonymous2', 'Second By George Hufford': 'George Hufford', 'Second By Getzler': 'Abraham Getzler', 'Second By Gladstein / Tester': 'David Gladstein', 'Second By Graaskamp & Katzen': 'Jim Graaskamp & Ken Katzen', 'Second By Grofman': 'Bernard Grofman', 'Second By Hall': 'James E. Hall', 'Second By Harrington': 'Paul D. Harrington', 'Second By Hollander': 'Howard R. Hollander', 'Second By Hotz': 'Roger Hotz', 'Second By Jones': 'Martyn Jones', 'Second By Joss': 'Johann Joss', 'Second By Kluepfel': 'Charles Kluepfel', 'Second By Lefevre': 'George Lefevre', 'Second By Leyland': 'Robert A. Leyland', 'Second By Leyvraz': 'Fransois Leyvraz', 'Second By Mcgurrin': 'Michael F. McGurrin', 'Second By Mikkelson': 'Ray Mikkelson', 'Second By Newman': 'Steve Newman', 'Second By Nussbacher': 'Henry Nussbacher', 'Second By Nydegger': 'Rudy Nydegger', 'Second By Pebly': 'Robert Pebly', 'Second By Pinkley': 'Jonathan Pinkley', 'Second By Quayle': 'Stanley F. Quayle', 'Second By Rabbie': 'Harold Rabbie', 'Second By Richard Hufford': 'Richard Hufford', 'Second By Robert Adams': 'Robert Adams', 'Second By Robertson': 'William H. Robertson', 'Second By Rowsam': 'Glen Rowsam', 'Second By Schurmann': 'E. E. H. Schurmann', 'Second By Smith': 'David A. Smith', 'Second By Smoody': 'Rik Smoody', 'Second By Snodgrass': 'Gene Snodgrass', 'Second By Tideman & Chieruzzi': 'T.Nicolaus Tideman & Paula Chieruzzi', 'Second By Tranquilizer': 'Craig Feathers', 'Second By Weiderman': 'Nelson Weiderman', 'Second By Weiner': 'Herb Weiner', 'Second By White': 'Edward C. White Jr', 'Second By Wiliam Adams': 'William Adams', 'Second By Yamachi': 'Brian Yamachi', 'Second By Zimmerman': 'George Zimmerman', 'Second Chance': 'Benelliott', 'Second Chance Attariq Version': 'M. A. M. Attariq', 'Self Steem': 'L. C. Andre & P. Honovan & T. Felipe & G. Frederico', 'Self-Love A-Learner': 'M. A. M. Attariq', 'Short Mem': 'L. C. Andre & P. Honovan & T. Felipe & G. Frederico', 'Signaling Cheater': 'Eckhart Arnold', 'Simple ANN': 'M. A. M. Attariq', 'Simple Exploiter': 'M. A. M. Attariq', 'Simple Identity ChecK': 'Red75', 'Simpleton': 'Eckhart Arnold', 'Skinner': 'M. A. M. Attariq', 'Slanger': 'M. A. M. Attariq', 'Slow Tit For Tat': "LIFL(Laboratoire d'Informatique Fondamentale de Lille)", 'Smart': 'Sven Sommer', 'Smart Cat': 'M. A. M. Attariq & Syarif', 'Smart Generous Tit For Tat': 'M. A. M. Attariq', 'Smart Go By Majority': 'M. A. M. Attariq', 'Smart Snake': 'M. A. M. Attariq', 'Smart-Clever': 'M. A. M. Attariq', 'Smarter Tat For Tit': 'Eckhart Arnold', 'Smooth Grudger': 'M. A. M. Attariq', 'Smooth Tit For Tat': 'M. A. M. Attariq', 'Sneaky Tit For Tat': 'Karol Langner', 'Social Average': 'Chat-GPT', 'Social Engineering': 'M. A. M. Attariq', 'Socio': 'M. A. M. Attariq', 'Soft Grim': 'Eckhart Arnold', 'Soft Grudger': 'J. Li & P. Hingston & G. Kendall', 'Solution B1': 'D. Ashlock & J. A. Brown & P. Hingston', 'Solution B5': 'D. Ashlock & J. A. Brown & P. Hingston', 'Sort': 'M. A. M. Attariq', 'Spectrum Attacker': 'M. A. M. Attariq', 'Spiteful CC': 'P. Mathieu & J. Delahaye', 'Spiteful Tit For Tat': "LIFL(Laboratoire d'Informatique Fondamentale de Lille)", 'Split Brain Syndrome': 'Dola', 'Springter': 'M. A. M. Attariq', 'Stabilize': 'Paul Boehm', 'Stalker': 'L. C. Andre & P. Honovan & T. Felipe & G. Frederico', 'Star S': 'Gopal Ramchurn', 'Star SL': 'Gopal Ramchurn', 'Star SN': 'Gopal Ramchurn', 'Star Slave 1': 'Gopal Ramchurn', 'Star Slave 10': 'Gopal Ramchurn', 'Star Slave 2': 'Gopal Ramchurn', 'Star Slave 3': 'Gopal Ramchurn', 'Star Slave 4': 'Gopal Ramchurn', 'Star Slave 5': 'Gopal Ramchurn', 'Star Slave 6': 'Gopal Ramchurn', 'Star Slave 7': 'Gopal Ramchurn', 'Star Slave 8': 'Gopal Ramchurn', 'Star Slave 9': 'Gopal Ramchurn', 'Statistical Strategy': 'd0minik2', 'Stefan Pennartz': 'Stefan Pennartz', 'Stochastic Cooperator': 'C. Adami & A. Hintze', 'Stochastic WSLS': 'Marc Harper', 'Stoic Mirror': 'Gemini', 'Stoicalist': 'M. A. M. Attariq & Gemini', 'Strategic Commitment': 'Thomas Schelling', 'Strategist': 'Explerify.com', 'Stupidiot': 'M. A. M. Attariq', 'Sugar Strategy / Treasure Hunt': 'D. Ashlock & W. Ashlock', 'Sumobox Fighter': 'M. A. M. Attariq', 'Sun Tzu Bot': 'M. A. M. Attariq', 'Suspicious Alternator': 'Benjamin M. Zagorsky & Johannes G. Reiter & Krishnendu Chatterjee & Martin A. Nowak', 'Suspicious Tit For Tat': 'C. Hilbe & M. A. Nowak & A. Traulsen', 'Sweet Heart': 'DeepSeek', 'Syifa': 'M. A. M. Attariq & Syifa', 'TF 1': 'Marc Harper', 'TF 2': 'Marc Harper', 'TF 3': 'Marc Harper', 'TF2T & 2TFT': 'M. A. M. Attariq', 'TOM Level 1 / Theory Of Mind Level 1': 'M. A. M. Attariq', 'TOM Level 2 / Theory Of Mind Level 2': 'M. A. M. Attariq', 'TOM Level 3 / Theory Of Mind Level 3': 'M. A. M. Attariq', 'TPMS / Tree Predictive Matrix Strategy': 'M. A. M. Attariq', 'Tages': 'Marco Gaudesi & Elio Piccolo & Giovanni Squillero & Alberto Paolo Tonda', 'Tau': 'M. A. M. Attariq', 'Teacher Pee While Standing, Student Pee While Running': 'M. A. M. Attariq', 'Tester Eckhart Arnold Version': 'Eckhart Arnold', 'The Calculated Mirror With A Fuse': 'Gemini', 'The Cave Breaker': 'M. A. M. Attariq & Gemini', 'The Nasty A': 'M. A. M. Attariq', 'Theseus': 'M. A. M. Attariq', 'Three Times Out': 'Philip Seehausen', 'Threshold Punisher': 'Copilot', 'Thresholded Trust Bot': 'M. A. M. Attariq', 'Thue Morse': 'Geraint Palmer', 'Thue Morse Inverse': 'Geraint Palmer', 'Thumper': 'D. Ashlock & E. Y. Kim', 'Tiny Brain': 'M. A. M. Attariq', 'Tip For Tap': 'M. A. M. Attariq', 'Tit For 3 Tats': 'Yaroslav Rosokha & Julian Romero', 'Tit For Increasing Tat': 'Claude', 'Tit For Tat But Not Too Forgiving': 'M. A. M. Attariq', 'Tit For Tat Killer': 'M. A. M. Attariq', 'Toxic Mirror': 'DeepSeek', 'Trader': 'M. A. M. Attariq & Gemini', 'Tranquilizer Eckhart Arnold Version': 'Eckhart Arnold', 'Tri-Brain': 'M. A. M. Attariq', 'Tricky Cooperator': 'Karol Langner', 'Tricky Defector': 'Karol Langner', 'Tricky Level Punisher': 'Eckhart Arnold', 'Trust Ledger': 'Chat-GPT', 'Trust Score Sigmoid': 'Claude', 'Trusty': 'Foomii', 'Turan': 'Marco Gaudesi & Elio Piccolo & Giovanni Squillero & Alberto Paolo Tonda', 'Tweedledee': 'Eckhart Arnold', 'Tweedledum': 'Eckhart Arnold', 'Tweetypie': 'Eckhart Arnold', 'Unpredictable Tit For Tat': 'M. A. M. Attariq', 'Uranium': 'M. A. M. Attariq', 'Usually Cooperates': 'D. Ashlock & E. Y. Kim & W. Ashlock', 'Usually Defects': 'D. Ashlock & E. Y.Kim & W. Ashlock', 'Vandal': 'Foomii', 'Vengeful': 'D. Ashlock', 'Vengeful Cheater': 'Vaniver', 'Verity': 'M. A. M. Attariq & Gemini', 'Very Bad': 'L. C. Andre & P. Honovan & T. Felipe & G. Frederico', 'WHAgent': 'remcofl', 'Whale': 'M. A. M. Attariq & DeepSeek', 'William': 'M. A. M. Attariq', 'Willing': 'P. Van Den Berg & F. J. Weissing', 'Win-Shift, Lose-Stay': 'J. Li & P. Hingston & G. Kendall', 'Winner 12': 'P. Mathieu & J. Delahaye', 'Winner 21': 'P. Mathieu & J. Delahaye', 'Worse & Worse / Worse & Worse 1': "LIFL(Laboratoire d'Informatique Fondamentale de Lille)", 'Worse & Worse 2': "LIFL(Laboratoire d'Informatique Fondamentale de Lille)", 'Worse & Worse 3': "LIFL(Laboratoire d'Informatique Fondamentale de Lille)", 'YOUR CATEGORY': 'M. A. M. Attariq', 'Your Choice Or My Choice': 'M. A. M. Attariq', 'ZD-2026 / Zero Determinant 2026': 'M. A. M. Attariq', 'ZD-G / Zero Determinant Generous': "Steven Kuhn, But Implemented In this Program Using C. Hilbe, M. A. Nowak, & A. Traulsen 's Model", 'ZD-GTFT / Zero Determinant Generous Tit For Tat': "A. J. Stewart & J. B. Plotkin, But Implemented In this Program Using C. Hilbe, M. A. Nowak, & A. Traulsen 's Model", 'ZD-M2 / Zero Determinant Memory 2': 'Marc Harper', 'ZD-MD / Zero Determinant Middle': 'M. A. M. Attariq', 'ZD-MS / Zero Determinant Mischief': "Lars Roemheld, But Implemented In this Program Using C. Hilbe, M. A. Nowak, & A. Traulsen 's Model", 'ZD-PW2026 / Zero Determinant Psychological War 2026': 'Gemini', 'ZD-Q / Zero Determinant Equalizer': "William H. Press & Freeman Dyson, But Implemented In this Program Using C. Hilbe, M. A. Nowak, & A. Traulsen 's Model", 'ZD-S / Zero Determinant Set': "Steven Kuhn, But Implemented In this Program Using C. Hilbe, M. A. Nowak, & A. Traulsen 's Model", 'ZD-X / Zero Determinant Extortion': "William H. Press & Freeman Dyson, But Implemented In this Program Using C. Hilbe, M. A. Nowak, & A. Traulsen 's Model", 'ZD-X A. J. Stewart & J. B. Plotkin Version / Zero Determinant Extortion A. J. Stewart & J. B. Plotkin Version': "A. J. Stewart & J. B. Plotkin, But Implemented In this Program Using C. Hilbe, M. A. Nowak, & A. Traulsen 's Model", 'ZD-X Lars Roemheld Version / Zero Determinant Extortion Lars Roemheld Version': "Lars Roemheld, But Implemented In this Program Using C. Hilbe, M. A. Nowak, & A. Traulsen 's Model", 'ZD-X Marc Harper Version / Zero Determinant Extortion Marc Harper Version': 'Marc Harper', 'ZD-X Steven Kuhn Version / Zero Determinant Extortion Steven Kuhn Version': "Steven Kuhn, But Implemented In this Program Using C. Hilbe, M. A. Nowak, & A. Traulsen 's Model", 'Zero Day': 'M. A. M. Attariq', 'Zero Input Apery': 'M. A. M. Attariq', 'Zero Input Champernowne': 'M. A. M. Attariq', 'Zero Input Euler': 'M. A. M. Attariq', 'Zero Input Euler-Mascheroni': 'M. A. M. Attariq', 'Zero Input Phi': 'M. A. M. Attariq', 'Zero Input Pi': 'M. A. M. Attariq', 'Zero Input Pythagoras': 'M. A. M. Attariq', 'Zero Input Tau': 'M. A. M. Attariq', 'ZeroResp': 'Evreu1pro', 'ZeroResp V5.3': 'Evreu1pro'}
strategies_category = {'2 Tits For 2 Tats Eckhart Arnold Version': 'NASTY', '2 Tits For Tat': 'NICE', '3 Seconds Mirror': 'NICE', '3-Tree Of Prediction': 'NICE', '4-Duality Optimizer 1': 'NICE', 'A-A&C / A-Actor&Critic': 'NASTY', 'ALLC OR ALLD': 'NASTY', 'AMERICA! / Freedoom / USA': 'NASTY', 'AON2': 'NICE', 'AQUA': 'NICE', 'AZD / Adaptive Zero Determinant': 'NICE', 'Absurdistan 1': 'NICE', 'Acquaintance & Trust': 'NICE', 'Adaptable Game type Strategy': 'NICE', 'Adaptive': 'NASTY', 'Adaptive Cooperator': 'NICE', 'Adaptive Hhhpraise Version': 'NICE', 'Adaptive Pavlov 2006': 'NICE', 'Adaptive Pavlov 2011': 'NICE', 'Adaptive Tit For Tat': 'NICE', 'Adaptive Tit For Tat Attariq Version': 'NICE', 'Adaptor Brief': 'NASTY', 'Adaptor Long': 'NASTY', 'Aggravater': 'NASTY', 'Alan': 'NICE', 'Alcohol / Ethanol': 'NASTY', 'Alexei / End Game Cheater': 'NASTY', 'Alternating Strategy': 'NASTY', 'Alternator / Second By Duisman': 'NASTY', 'Alternator Hunter': 'NICE', 'Always Cooperate / Cooperator / Dove / Saint': 'NICE', 'Always Defect / Defector / Hawk / Shark': 'NASTY', 'Always Out / Outtor': 'NICE', 'Amazon / Piraha': 'NICE', 'Analogy': 'NASTY', 'Anatol': 'NICE', 'Angry Tit For Tat': 'NICE', 'Anti Bourgeois': 'NASTY', 'Anti Cycler': 'NASTY', 'Anti Noise Tit For Tat': 'NICE', 'Anti Tit For Tat / Psycho': 'NASTY', 'Apery': 'NASTY', 'Appeaser': 'NICE', 'Arrogant Q-Learner': 'NASTY', "Attariq's Cheating Matching Pennies Strategy": 'NASTY', 'Average Copier': 'NASTY', 'BBE / Burn Both Ends': 'NASTY', 'Back Stabber': 'NICE', 'Backrooms': 'NASTY', 'Backward Induction': 'NASTY', 'Bahlil': 'NASTY', 'Bandit': 'NASTY', 'Bandit UCB': 'NASTY', 'Bayes Attariq Version': 'NICE', 'Bayes Foomii Version': 'NICE', 'Bayesian Forgiver': 'NICE', 'Benjo': 'NICE', 'Beta Tester': 'NASTY', 'Better & Better': 'NASTY', 'Bluff': 'NASTY', 'Bluff Master': 'NASTY', 'Boltzmann Brain': 'NASTY', 'Bookend Cooperator': 'NASTY', 'Bourgeois': 'NICE', 'Boxer': 'NICE', 'Bros Mind': 'NICE', 'Bully / Reverse Tit For Tat': 'NASTY', 'Burn The Bridge Strategy': 'NASTY', 'Bush Mosteller': 'NASTY', 'Buzzer': 'NASTY', 'CC-L / Cosmos-Chaos Learner': 'NASTY', 'CS / Collective Strategy': 'NASTY', 'Caerbannog': 'NASTY', 'Calamity': 'NASTY', 'Calculator': 'NASTY', 'Calculator The Nerd Of AI Version': 'NICE', 'Capitalist': 'NASTY', 'Capri': 'NICE', 'Cautious Q-Learner': 'NASTY', 'Caveman': 'NASTY', 'Cerebrum': 'NICE', 'Champernowne': 'NASTY', 'Charity': 'NICE', 'Chatter': 'NICE', 'Chimera': 'NICE', 'Clement Coldridge': 'NICE', 'Coin': 'NASTY', 'Collatz': 'NASTY', 'Communalist': 'NICE', 'Continuous Q-Learner Avg': 'NASTY', 'Continuous Q-Learner Max': 'NASTY', 'Contrite Grudger': 'NICE', 'Contrite Tit For Tat': 'NICE', 'Control 1': 'NASTY', 'Control 10': 'NASTY', 'Control 11': 'NASTY', 'Control 2': 'NICE', 'Control 3': 'NASTY', 'Control 4': 'NASTY', 'Control 5': 'NASTY', 'Control 6': 'NICE', 'Control 7': 'NASTY', 'Control 8': 'NASTY', 'Control 9': 'NICE', 'Cooperate ISO': 'NICE', 'Cooperate Once': 'NASTY', 'Cooperator Hunter': 'NASTY', 'Crabby': 'NICE', 'Crabby Attariq Version': 'NASTY', 'Cruelity': 'NICE', 'Cult Bishop A': 'NASTY', 'Cult Bishop B': 'NASTY', 'Cult Citizen A': 'NASTY', 'Cult Citizen B': 'NASTY', 'Cult Citizen C': 'NASTY', 'Cult Leader': 'NASTY', 'Curiosity': 'NICE', 'Cycle Hunter': 'NASTY', 'Cycler CCCCCD': 'NASTY', 'Cycler CCCD': 'NASTY', 'Cycler CCCDCD': 'NASTY', 'Cycler CCD': 'NASTY', 'Cycler DC': 'NASTY', 'Cycler DCCCC': 'NASTY', 'Cycler DDC': 'NASTY', 'DBS / Derived Belief Strategy': 'NICE', 'DDoS': 'NICE', 'Darwin': 'NICE', 'Deadlock Breaker': 'NICE', 'Decay Given': 'NICE', 'Defect Once': 'NASTY', 'Defector Hunter': 'NICE', 'Delayed AON1': 'NICE', 'Desire': 'NICE', 'Desprate': 'NASTY', 'Detective': 'NASTY', 'Devil Staircase': 'NICE', 'Diplomat': 'NICE', 'Dont Bite The Hand That Feeds You': 'NICE', 'Double Crosser': 'NICE', 'Double Q-Learner': 'NASTY', 'Double Resurrection': 'NASTY', 'Doubler': 'NICE', 'Downing Benjjo Version': 'NICE', 'Downing Eckhart Arnold Version': 'NASTY', 'Dumpty 10': 'NASTY', 'Dyna Q-Learner': 'NASTY', 'Dynamic 2 Tits For Tat': 'NICE', 'Dynocat': 'NASTY', 'EGOist': 'NASTY', 'Easy Go / Fool Me Forever / Reverse Grudger': 'NASTY', 'Echo Adaptive': 'NICE', 'Echo Fisher': 'NASTY', 'Egoist A-Learner': 'NICE', 'Eliza / Therapist': 'NICE', 'Emotional Tit For Tat': 'NICE', 'Entropy Sentinel': 'NASTY', 'Envy': 'NICE', 'Epsilon-Greedy': 'NASTY', 'Eric': 'NICE', 'Eugine Nier': 'NASTY', 'Euler': 'NASTY', 'Euler-Mascheroni': 'NASTY', 'Eventual Cycle Hunter': 'NICE', 'Evil Alliance': 'NASTY', 'Evolvable Cycler Attariq Version': 'NASTY', 'Evolved FSM 16': 'NICE', 'Evolved FSM 16 Noise 05': 'NICE', 'Evolved FSM 4': 'NICE', 'Evolved FSM 6': 'NASTY', 'Evolved HMM 5': 'NICE', 'Evolved Looker Up 1_1_1': 'NICE', 'Evolved Looker Up 2_2_2': 'NICE', 'Exploratory Lenient Grim 2': 'NASTY', 'Exploratory Tit For 3 Tats': 'NASTY', 'FAWS': 'NASTY', 'False Cooperator': 'NASTY', 'Falsity': 'NICE', 'Female / Woman': 'NASTY', 'Fibonacci': 'NASTY', 'Fictitious Play': 'NASTY', 'Finger Print': 'NASTY', 'Firm But Fair / Firm For Tat': 'NICE', 'First By Davis': 'NICE', 'First By Downing': 'NASTY', 'First By Downing, But Get Revised By Axelrod': 'NICE', 'First By Feld': 'NASTY', 'First By Graaskamp / Second By Mauk': 'NASTY', 'First By Grofman': 'NICE', 'First By Grudger / Grim Trigger': 'NICE', 'First By Joss / Naive Prober': 'NASTY', 'First By Named Withheld': 'NASTY', 'First By Nydegger': 'NICE', 'First By Shubik': 'NICE', 'First By Stein & Rapoport': 'NASTY', 'First By Tideman & Chieruzzi': 'NICE', 'First By Tit For Tat': 'NICE', 'First By Tullock': 'NASTY', 'Fixed Pattern': 'NASTY', 'Focal Point Strategy / Schelling Point Strategy': 'NASTY', 'Fool Me Once': 'NICE', 'Forest-RND': 'NICE', 'Forest-XG': 'NICE', 'Forgetable Gradual': 'NICE', 'Forgetful Fool Me Once': 'NICE', 'Forgetful Grudger': 'NICE', 'Forgetful Majority Bot': 'NICE', 'Forgiver': 'NICE', 'Forgiver Benjamin M. Zagorsky & Johannes G. Reiter & Krishnendu Chatterjee & Martin A. Nowak Version / S14': 'NICE', 'Forgiving': 'NICE', 'Forgiving Fate': 'NICE', 'Forgiving Grudger': 'NICE', 'Forgiving Tit For Tat': 'NICE', 'Fortress 3': 'NASTY', 'Fortress 4': 'NASTY', 'Frankl': 'NICE', 'Free Rider': 'NASTY', 'Frequency Analyzer': 'NICE', 'Freud': 'NICE', 'Friendly Human': 'NICE', 'GT-E / Game Theory Explorer': 'NICE', 'Gambler Attariq Version': 'NASTY', 'Gaslighter': 'NASTY', 'Geller': 'NICE', 'Geller Cooperator': 'NICE', 'Geller Defector': 'NICE', 'Gen 1_64_200_30_0.3_A': 'NASTY', 'Gen 2_64_200_30_0.3_A': 'NASTY', 'Gen 3_64_200_30_0.3_A': 'NASTY', 'Gen 4_64_200_30_0.3_A': 'NASTY', 'Gen 5_64_200_30_0.3_A': 'NASTY', 'Generic Downing': 'NASTY', 'Generic Downing Friendly': 'NASTY', 'Generous Tit For Tat': 'NICE', 'Generous Tit For Tat Axelrod Project Contributor Team Version': 'NICE', 'Generous Tit For Tat Benjjo Version': 'NICE', 'Generous Tit For Tat Eckhart Arnold Version 1': 'NICE', 'Generous Tit For Tat Eckhart Arnold Version 2': 'NICE', 'Generous Tit For Tat Less Wrong Version': 'NICE', 'Genetic Algo 1': 'NICE', 'Genetic Algo 2': 'NICE', 'George': 'NASTY', 'Go By Majority / Second By Grisell / Soft Go By Majority': 'NICE', 'Go By Minority': 'NASTY', 'Gold Digger': 'NASTY', 'Golden Mean': 'NICE', 'Good': 'NICE', 'GossipAgent': 'NASTY', 'Graaskamp Benjjo Version': 'NASTY', 'Gradual': 'NICE', 'Gradual Cristal Version': 'NICE', 'Gradual Killer': 'NASTY', 'Grateful / S9': 'NASTY', 'Greedy': 'NICE', 'Grudger Alternator': 'NICE', 'Grumpy': 'NICE', 'HMM Player': 'NASTY', 'Half Sin Square': 'NICE', 'Handshake': 'NASTY', 'Hard Go By Majority': 'NICE', 'Hard Prober': 'NASTY', 'Hard Tit For 2 Tats': 'NICE', 'Hard Tit For Tat': 'NICE', 'Hesitant Q-Learner': 'NASTY', 'Hill Climb': 'NASTY', 'Hill Climb Attariq Version': 'NASTY', 'Hopeless': 'NASTY', 'Hot Coffee': 'NICE', 'Humpty 10': 'NASTY', 'Hungry': 'NASTY', 'ISO': 'NICE', 'Inequality / SOVIET UNION! / USSR': 'NASTY', 'Inequity Averse Q-Learner': 'NASTY', 'Inverse': 'NICE', 'Inverse Punisher': 'NICE', 'Is There Something New?': 'NICE', 'Isolated Clan': 'NASTY', 'Isolated Tit For Tat': 'NICE', 'Japan Empire Emperor': 'NICE', 'Japan Empire Soldier 1': 'NICE', 'Japan Empire Soldier 10': 'NICE', 'Japan Empire Soldier 2': 'NICE', 'Japan Empire Soldier 3': 'NICE', 'Japan Empire Soldier 4': 'NICE', 'Japan Empire Soldier 5': 'NICE', 'Japan Empire Soldier 6': 'NICE', 'Japan Empire Soldier 7': 'NICE', 'Japan Empire Soldier 8': 'NICE', 'Japan Empire Soldier 9': 'NICE', 'Javelin': 'NICE', 'Jem': 'NASTY', 'Jester': 'NASTY', 'Jesus Petry': 'NASTY', 'Josuah': 'NASTY', 'Jung': 'NICE', 'Karen': 'NASTY', 'Keyman': 'NASTY', 'Killer': 'NASTY', 'King Of 48 Laws Of Power': 'NICE', 'Knowledgeable Worse & Worse': 'NASTY', 'Konflikt Blue': 'NASTY', 'Konflikt Green': 'NICE', 'L-BODOR / Learning Based On Derivative Of Reward': 'NICE', 'LCG': 'NASTY', 'Laran': 'NASTY', 'Laws': 'NICE', 'Learner': 'NICE', 'Lenient Grim 2': 'NICE', 'Lenient Grim 3': 'NICE', 'Less Grim': 'NICE', 'Level Punisher': 'NICE', 'Liar Person': 'NASTY', 'Lighty-Darky': 'NASTY', 'Limited Retaliate': 'NICE', 'Limited Retaliate 2': 'NICE', 'Limited Retaliate 3': 'NICE', 'Line': 'NICE', 'Little Grudger': 'NICE', 'Local Copycat': 'NASTY', 'Long Horse': 'NASTY', 'Long Term Strategist': 'NICE', 'Longterm Tit For Tat': 'NICE', 'Look Ahead / Look Up': 'NASTY', 'Loyal': 'NICE', 'MARS / Mimicry And Relative Similarity': 'NICE', 'MCMC / Markov Chain Monte Carlo': 'NICE', 'MEM2': 'NICE', 'MENACE': 'NICE', 'MSAgent': 'NASTY', 'Mafia A': 'NASTY', 'Mafia B': 'NASTY', 'Mafia C': 'NASTY', 'Mafia D': 'NASTY', 'Majapahit': 'NICE', 'Male / Man': 'NICE', 'Malthrin': 'NASTY', 'Man In The Middle': 'NASTY', 'Manipulator': 'NASTY', 'Manipulity': 'NASTY', 'Markov': 'NICE', 'Matcher': 'NICE', 'Math Constant Hunter': 'NICE', 'Memory 1': 'NASTY', 'Memory Decay': 'NICE', 'Mensa': 'NASTY', 'Meta Hunter': 'NASTY', 'Meta Hunter Aggressive': 'NASTY', 'Meta Majority': 'NICE', 'Meta Minority': 'NASTY', 'Meta Mixer': 'NASTY', 'Meta Winner': 'NASTY', 'Meta Winner Ensemble': 'NASTY', 'Michaelos': 'NASTY', 'Minimax Strategy': 'NASTY', 'Mirror': 'NICE', 'Mixed Strategy': 'NASTY', 'Modal Defector': 'NASTY', 'Modal Tit For Tat': 'NICE', "Modified Primer's Strategy": 'NASTY', 'Momentum': 'NICE', 'Monkey See, Monkey Do': 'NASTY', 'Mosquito': 'NASTY', 'MrX': 'NASTY', 'Mystical Logic': 'NASTY', 'N Pavlov': 'NASTY', 'N Tits For M Tats': 'NICE', 'Naive Q-Learner': 'NASTY', 'Nash': 'NICE', 'Nasty Tit For Tat': 'NASTY', 'Negation': 'NASTY', 'Neil A': 'NICE', 'Newton': 'NASTY', 'Nice Meta Winner': 'NICE', 'Nice Meta Winner Ensemble': 'NICE', 'Nydegger Benjjo Version': 'NICE', 'Observant': 'NICE', 'Odd-Even Go By Majority': 'NICE', 'Odysseus / Ὀδυσσεύς': 'NICE', 'Omega Tit For Tat': 'NICE', 'Once Bitten': 'NICE', 'Opportunist': 'NASTY', 'Opposite Grudger': 'NASTY', 'Out For Tat': 'NICE', 'PIN': 'NASTY', 'PSO Gambler 1_1_1': 'NICE', 'PSO Gambler 2_2_2': 'NICE', 'PSO Gambler 2_2_2 Noise 05': 'NICE', 'PSO Gambler Mem 1': 'NICE', 'Paradoxic Grim / S21': 'NASTY', 'Parameterized Tit For Tat': 'NASTY', 'Paranoid': 'NASTY', 'Passive Tit For Tat': 'NICE', 'Patient Cooperator': 'NICE', 'Pattern Reader': 'NICE', 'Patterned Adaptive Fortress': 'NICE', 'Patterned Adaptive Fortress 2': 'NICE', 'Pavlov / Win-Stay, Lose-Shift': 'NICE', 'Pavlov D / Suspicious Pavlov': 'NASTY', 'Pavlov Tester': 'NASTY', 'Pavolovo': 'NASTY', 'Pawlow': 'NICE', 'Payoff-Dominance Strategy': 'NICE', 'Perceptron': 'NASTY', 'Phi': 'NASTY', 'Pi': 'NASTY', 'Prabowo': 'NASTY', 'Praedator': 'NASTY', 'Predator': 'NASTY', "Primer's Strategy": 'NASTY', 'Probable Coop': 'NASTY', 'Probe & Exploit': 'NASTY', 'Probe & Punish': 'NICE', 'Prober': 'NASTY', 'Prober 2': 'NASTY', 'Prober 3': 'NASTY', 'Prober 4': 'NASTY', 'Prober Abdullah-arif-swe Version': 'NASTY', 'Procastinater': 'NICE', 'Psycho Attariq Version': 'NICE', 'Pun 1': 'NASTY', 'Punisher': 'NICE', 'Pythagoras': 'NASTY', 'Q-Learning A-Learner': 'NICE', 'REINFORCE': 'NASTY', 'RLAgent': 'NASTY', 'RLGossipAgent': 'NASTY', 'RNN & RTRL': 'NASTY', 'Racister': 'NASTY', 'Rack Block Shooter': 'NASTY', 'Raider': 'NASTY', 'Random': 'NASTY', 'Random Hunter': 'NICE', 'Random Tit For Tat': 'NASTY', "Random's Exploiter 1": 'NICE', "Random's Exploiter 2": 'NICE', "Random's Exploiter 3": 'NICE', "Random's Exploiter 4": 'NICE', 'Reactive Player': 'NASTY', 'Rememberable Grudger': 'NICE', 'Remorseful Prober': 'NASTY', 'Resurrection': 'NICE', 'Retaliate': 'NICE', 'Retaliate 2': 'NICE', 'Retaliate 3': 'NICE', 'Rice Field': 'NASTY', 'Ripoff': 'NASTY', 'Risk-Dominance Strategy': 'NASTY', 'Risky Q-Learner': 'NASTY', 'Rogers & Maslow': 'NICE', 'Rover': 'NASTY', 'Rwallace': 'NASTY', 'S11': 'NASTY', 'S12 / Suspicious Forgiver': 'NASTY', 'S18': 'NASTY', 'S19': 'NASTY', 'S2': 'NASTY', 'S23': 'NASTY', 'S24': 'NASTY', 'S6': 'NASTY', 'S7': 'NASTY', 'SURPRISE ATTACK!': 'NASTY', 'Sample / Second By Tit For 2 Tats': 'NICE', 'Sarsa Q-Learner': 'NASTY', 'Schwoba Sen Gnitz': 'NICE', 'Scorched Earth': 'NICE', 'Score Equalizer': 'NICE', 'Second By Almy': 'NASTY', 'Second By Ambuehl & Hickey': 'NICE', 'Second By Anderson': 'NASTY', 'Second By Appold': 'NICE', 'Second By Batell': 'NICE', 'Second By Black': 'NICE', 'Second By Borufsen': 'NICE', 'Second By Cave': 'NICE', 'Second By Champion': 'NICE', 'Second By Colbert': 'NASTY', 'Second By Dawes & Batell': 'NICE', 'Second By Eatherley': 'NICE', 'Second By Falk & Langsted': 'NICE', 'Second By Feld': 'NASTY', 'Second By Friedland': 'NASTY', 'Second By Friedland, But Get Revised By Axelrod': 'NASTY', "Second By GRASR's Author": 'NASTY', 'Second By George Hufford': 'NASTY', 'Second By Getzler': 'NICE', 'Second By Gladstein / Tester': 'NASTY', 'Second By Graaskamp & Katzen': 'NICE', 'Second By Grofman': 'NICE', 'Second By Hall': 'NASTY', 'Second By Harrington': 'NASTY', 'Second By Hollander': 'NASTY', 'Second By Hotz': 'NASTY', 'Second By Jones': 'NICE', 'Second By Joss': 'NASTY', 'Second By Kluepfel': 'NICE', 'Second By Lefevre': 'NICE', 'Second By Leyland': 'NASTY', 'Second By Leyvraz': 'NICE', 'Second By Mcgurrin': 'NASTY', 'Second By Mikkelson': 'NICE', 'Second By Newman': 'NICE', 'Second By Nussbacher': 'NICE', 'Second By Nydegger': 'NICE', 'Second By Pebly': 'NICE', 'Second By Pinkley': 'NICE', 'Second By Quayle': 'NICE', 'Second By Rabbie': 'NASTY', 'Second By Richard Hufford': 'NASTY', 'Second By Robert Adams': 'NICE', 'Second By Robertson': 'NASTY', 'Second By Rowsam': 'NICE', 'Second By Schurmann': 'NICE', 'Second By Smith': 'NASTY', 'Second By Smoody': 'NASTY', 'Second By Snodgrass': 'NASTY', 'Second By Tideman & Chieruzzi': 'NICE', 'Second By Tranquilizer': 'NASTY', 'Second By Weiderman': 'NICE', 'Second By Weiner': 'NICE', 'Second By White': 'NICE', 'Second By Wiliam Adams': 'NICE', 'Second By Yamachi': 'NICE', 'Second By Zimmerman': 'NICE', 'Second Chance': 'NASTY', 'Second Chance Attariq Version': 'NICE', 'Self Steem': 'NASTY', 'Self-Love A-Learner': 'NICE', 'Short Mem': 'NICE', 'Signaling Cheater': 'NASTY', 'Simple ANN': 'NASTY', 'Simple Exploiter': 'NASTY', 'Simple Identity ChecK': 'NASTY', 'Simpleton': 'NASTY', 'Skinner': 'NICE', 'Slanger': 'NICE', 'Slow Tit For Tat': 'NICE', 'Smart': 'NICE', 'Smart Cat': 'NICE', 'Smart Generous Tit For Tat': 'NICE', 'Smart Go By Majority': 'NICE', 'Smart Snake': 'NICE', 'Smart-Clever': 'NICE', 'Smarter Tat For Tit': 'NASTY', 'Smooth Grudger': 'NICE', 'Smooth Tit For Tat': 'NICE', 'Sneaky Tit For Tat': 'NASTY', 'Social Average': 'NASTY', 'Social Engineering': 'NASTY', 'Socio': 'NICE', 'Soft Grim': 'NICE', 'Soft Grudger': 'NICE', 'Solution B1': 'NASTY', 'Solution B5': 'NASTY', 'Sort': 'NICE', 'Spectrum Attacker': 'NASTY', 'Spiteful CC': 'NICE', 'Spiteful Tit For Tat': 'NICE', 'Split Brain Syndrome': 'NICE', 'Springter': 'NICE', 'Stabilize': 'NICE', 'Stalker': 'NICE', 'Star S': 'NASTY', 'Star SL': 'NASTY', 'Star SN': 'NASTY', 'Star Slave 1': 'NASTY', 'Star Slave 10': 'NASTY', 'Star Slave 2': 'NASTY', 'Star Slave 3': 'NASTY', 'Star Slave 4': 'NASTY', 'Star Slave 5': 'NASTY', 'Star Slave 6': 'NASTY', 'Star Slave 7': 'NASTY', 'Star Slave 8': 'NASTY', 'Star Slave 9': 'NASTY', 'Statistical Strategy': 'NASTY', 'Stefan Pennartz': 'NICE', 'Stochastic Cooperator': 'NASTY', 'Stochastic WSLS': 'NASTY', 'Stoic Mirror': 'NICE', 'Stoicalist': 'NICE', 'Strategic Commitment': 'NASTY', 'Strategist': 'NASTY', 'Stupidiot': 'NASTY', 'Sugar Strategy / Treasure Hunt': 'NASTY', 'Sumobox Fighter': 'NICE', 'Sun Tzu Bot': 'NICE', 'Suspicious Alternator': 'NASTY', 'Suspicious Tit For Tat': 'NASTY', 'Sweet Heart': 'NASTY', 'Syifa': 'NASTY', 'TF 1': 'NASTY', 'TF 2': 'NASTY', 'TF 3': 'NICE', 'TF2T & 2TFT': 'NICE', 'TOM Level 1 / Theory Of Mind Level 1': 'NASTY', 'TOM Level 2 / Theory Of Mind Level 2': 'NASTY', 'TOM Level 3 / Theory Of Mind Level 3': 'NICE', 'TPMS / Tree Predictive Matrix Strategy': 'NICE', 'Tages': 'NASTY', 'Tau': 'NASTY', 'Teacher Pee While Standing, Student Pee While Running': 'NICE', 'Tester Eckhart Arnold Version': 'NASTY', 'The Calculated Mirror With A Fuse': 'NASTY', 'The Cave Breaker': 'NASTY', 'The Nasty A': 'NASTY', 'Theseus': 'NICE', 'Three Times Out': 'NICE', 'Threshold Punisher': 'NICE', 'Thresholded Trust Bot': 'NICE', 'Thue Morse': 'NASTY', 'Thue Morse Inverse': 'NASTY', 'Thumper': 'NICE', 'Tiny Brain': 'NASTY', 'Tip For Tap': 'NICE', 'Tit For 3 Tats': 'NICE', 'Tit For Increasing Tat': 'NICE', 'Tit For Tat But Not Too Forgiving': 'NICE', 'Tit For Tat Killer': 'NASTY', 'Toxic Mirror': 'NASTY', 'Trader': 'NICE', 'Tranquilizer Eckhart Arnold Version': 'NASTY', 'Tri-Brain': 'NICE', 'Tricky Cooperator': 'NASTY', 'Tricky Defector': 'NASTY', 'Tricky Level Punisher': 'NICE', 'Trust Ledger': 'NICE', 'Trust Score Sigmoid': 'NASTY', 'Trusty': 'NICE', 'Turan': 'NASTY', 'Tweedledee': 'NICE', 'Tweedledum': 'NICE', 'Tweetypie': 'NASTY', 'Unpredictable Tit For Tat': 'NASTY', 'Uranium': 'NICE', 'Usually Cooperates': 'NICE', 'Usually Defects': 'NASTY', 'Vandal': 'NASTY', 'Vengeful': 'NICE', 'Vengeful Cheater': 'NASTY', 'Verity': 'NICE', 'Very Bad': 'NICE', 'WHAgent': 'NASTY', 'Whale': 'NICE', 'William': 'NICE', 'Willing': 'NASTY', 'Win-Shift, Lose-Stay': 'NASTY', 'Winner 12': 'NICE', 'Winner 21': 'NICE', 'Worse & Worse / Worse & Worse 1': 'NASTY', 'Worse & Worse 2': 'NASTY', 'Worse & Worse 3': 'NICE', 'YOUR CATEGORY': 'NICE', 'Your Choice Or My Choice': 'NICE', 'ZD-2026 / Zero Determinant 2026': 'NICE', 'ZD-G / Zero Determinant Generous': 'NICE', 'ZD-GTFT / Zero Determinant Generous Tit For Tat': 'NICE', 'ZD-M2 / Zero Determinant Memory 2': 'NASTY', 'ZD-MD / Zero Determinant Middle': 'NASTY', 'ZD-MS / Zero Determinant Mischief': 'NASTY', 'ZD-PW2026 / Zero Determinant Psychological War 2026': 'NICE', 'ZD-Q / Zero Determinant Equalizer': 'NASTY', 'ZD-S / Zero Determinant Set': 'NASTY', 'ZD-X / Zero Determinant Extortion': 'NASTY', 'ZD-X A. J. Stewart & J. B. Plotkin Version / Zero Determinant Extortion A. J. Stewart & J. B. Plotkin Version': 'NASTY', 'ZD-X Lars Roemheld Version / Zero Determinant Extortion Lars Roemheld Version': 'NASTY', 'ZD-X Marc Harper Version / Zero Determinant Extortion Marc Harper Version': 'NASTY', 'ZD-X Steven Kuhn Version / Zero Determinant Extortion Steven Kuhn Version': 'NASTY', 'Zero Day': 'NASTY', 'Zero Input Apery': 'NASTY', 'Zero Input Champernowne': 'NASTY', 'Zero Input Euler': 'NASTY', 'Zero Input Euler-Mascheroni': 'NASTY', 'Zero Input Phi': 'NASTY', 'Zero Input Pi': 'NASTY', 'Zero Input Pythagoras': 'NASTY', 'Zero Input Tau': 'NASTY', 'ZeroResp': 'NICE', 'ZeroResp V5.3': 'NASTY'}
strategies_coop_prob = {'2 Tits For 2 Tats Eckhart Arnold Version': 0.9459125710051864, '2 Tits For Tat': 1.0, '3 Seconds Mirror': 1.0, '3-Tree Of Prediction': 1.0, '4-Duality Optimizer 1': 1.0, 'A-A&C / A-Actor&Critic': 0.021391752577319588, 'ALLC OR ALLD': 0.7014816614039349, 'AMERICA! / Freedoom / USA': 0.4156521739130435, 'AON2': 1.0, 'AQUA': 1.0, 'AZD / Adaptive Zero Determinant': 1.0, 'Absurdistan 1': 1.0, 'Acquaintance & Trust': 1.0, 'Adaptable Game type Strategy': 1.0, 'Adaptive': 0.972291493488501, 'Adaptive Cooperator': 1.0, 'Adaptive Hhhpraise Version': 1.0, 'Adaptive Pavlov 2006': 1.0, 'Adaptive Pavlov 2011': 1.0, 'Adaptive Tit For Tat': 1.0, 'Adaptive Tit For Tat Attariq Version': 1.0, 'Adaptor Brief': 0.4701162147205313, 'Adaptor Long': 0.36043656207366986, 'Aggravater': 0.9849284099472494, 'Alan': 1.0, 'Alcohol / Ethanol': 0.9948757366128619, 'Alexei / End Game Cheater': 0.8943626206196038, 'Alternating Strategy': 0.0, 'Alternator / Second By Duisman': 0.5014113420579933, 'Alternator Hunter': 1.0, 'Always Cooperate / Cooperator / Dove / Saint': 1.0, 'Always Defect / Defector / Hawk / Shark': 0.0, 'Always Out / Outtor': 0.5, 'Amazon / Piraha': 1.0, 'Analogy': 0.009706381946129581, 'Anatol': 1.0, 'Angry Tit For Tat': 1.0, 'Anti Bourgeois': 0.0, 'Anti Cycler': 0.9116582186821144, 'Anti Noise Tit For Tat': 1.0, 'Anti Tit For Tat / Psycho': 0.00507227998985544, 'Apery': 0.004740459824602986, 'Appeaser': 1.0, 'Arrogant Q-Learner': 0.5046197583511016, "Attariq's Cheating Matching Pennies Strategy": 0.0, 'Average Copier': 0.9979118329466358, 'BBE / Burn Both Ends': 0.9051821679112921, 'Back Stabber': 1.0, 'Backrooms': 0.6848514369215781, 'Backward Induction': 0.0, 'Bahlil': 0.03041144901610018, 'Bandit': 0.48625592417061614, 'Bandit UCB': 0.014420555846879917, 'Bayes Attariq Version': 1.0, 'Bayes Foomii Version': 1.0, 'Bayesian Forgiver': 1.0, 'Benjo': 1.0, 'Beta Tester': 0.516845703125, 'Better & Better': 0.0937413840639647, 'Bluff': 0.9951585572500605, 'Bluff Master': 0.9484777517564403, 'Boltzmann Brain': 0.963315864697475, 'Bookend Cooperator': 0.998046875, 'Bourgeois': 1.0, 'Boxer': 1.0, 'Bros Mind': 1.0, 'Bully / Reverse Tit For Tat': 0.0, 'Burn The Bridge Strategy': 0.0041939711664482305, 'Bush Mosteller': 0.39702487817389076, 'Buzzer': 0.19221528111484862, 'CC-L / Cosmos-Chaos Learner': 0.5043041606886657, 'CS / Collective Strategy': 0.005339028296849973, 'Caerbannog': 0.8957452102513063, 'Calamity': 0.15675116399379202, 'Calculator': 0.09675759135357695, 'Calculator The Nerd Of AI Version': 1.0, 'Capitalist': 0.9818753020782987, 'Capri': 1.0, 'Cautious Q-Learner': 0.5314667988107037, 'Caveman': 0.802456232035537, 'Cerebrum': 1.0, 'Champernowne': 0.005219206680584551, 'Charity': 1.0, 'Chatter': 1.0, 'Chimera': 1.0, 'Clement Coldridge': 1.0, 'Coin': 0.5119402985074627, 'Collatz': 0.4442403568617161, 'Communalist': 1.0, 'Continuous Q-Learner Avg': 0.5085490962383976, 'Continuous Q-Learner Max': 0.3547321217842813, 'Contrite Grudger': 1.0, 'Contrite Tit For Tat': 1.0, 'Control 1': 0.12343423799582463, 'Control 10': 0.8805822847273624, 'Control 11': 0.9974240082431737, 'Control 2': 1.0, 'Control 3': 0.9950433705080545, 'Control 4': 0.9054660681278056, 'Control 5': 0.997589780670041, 'Control 6': 1.0, 'Control 7': 0.9983321420061949, 'Control 8': 0.8427461139896373, 'Control 9': 1.0, 'Cooperate ISO': 1.0, 'Cooperate Once': 0.005518763796909493, 'Cooperator Hunter': 0.019342359767891684, 'Crabby': 1.0, 'Crabby Attariq Version': 0.7727603787327021, 'Cruelity': 1.0, 'Cult Bishop A': 0.9896747547754259, 'Cult Bishop B': 0.9893983567452955, 'Cult Citizen A': 0.9893842887473461, 'Cult Citizen B': 0.9899749373433584, 'Cult Citizen C': 0.990528060620412, 'Cult Leader': 0.9897776641962689, 'Curiosity': 1.0, 'Cycle Hunter': 0.026219192448872573, 'Cycler CCCCCD': 0.8352997832891885, 'Cycler CCCD': 0.7514568026349125, 'Cycler CCCDCD': 0.6693609497347816, 'Cycler CCD': 0.6682341876322596, 'Cycler DC': 0.49831387808041505, 'Cycler DCCCC': 0.7982921763212555, 'Cycler DDC': 0.3313997477931904, 'DBS / Derived Belief Strategy': 1.0, 'DDoS': 1.0, 'Darwin': 1.0, 'Deadlock Breaker': 1.0, 'Decay Given': 1.0, 'Defect Once': 0.9945945945945946, 'Defector Hunter': 1.0, 'Delayed AON1': 1.0, 'Desire': 1.0, 'Desprate': 0.0029354207436399216, 'Detective': 0.014760147601476014, 'Devil Staircase': 1.0, 'Diplomat': 1.0, 'Dont Bite The Hand That Feeds You': 1.0, 'Double Crosser': 1.0, 'Double Q-Learner': 0.49987503124218946, 'Double Resurrection': 0.8353808353808354, 'Doubler': 1.0, 'Downing Benjjo Version': 1.0, 'Downing Eckhart Arnold Version': 0.021130480718436345, 'Dumpty 10': 0.7515585315169707, 'Dyna Q-Learner': 0.5268843014882382, 'Dynamic 2 Tits For Tat': 1.0, 'Dynocat': 0.8948655256723717, 'EGOist': 0.7569573283858998, 'Easy Go / Fool Me Forever / Reverse Grudger': 0.0, 'Echo Adaptive': 1.0, 'Echo Fisher': 0.5013385251886103, 'Egoist A-Learner': 1.0, 'Eliza / Therapist': 1.0, 'Emotional Tit For Tat': 1.0, 'Entropy Sentinel': 0.34438076252180416, 'Envy': 1.0, 'Epsilon-Greedy': 0.13625123639960435, 'Eric': 1.0, 'Eugine Nier': 0.8958694039079891, 'Euler': 0.004673989249824725, 'Euler-Mascheroni': 0.00550812448361333, 'Eventual Cycle Hunter': 1.0, 'Evil Alliance': 0.0, 'Evolvable Cycler Attariq Version': 0.32117150657731447, 'Evolved FSM 16': 1.0, 'Evolved FSM 16 Noise 05': 1.0, 'Evolved FSM 4': 1.0, 'Evolved FSM 6': 0.005198856251624643, 'Evolved HMM 5': 1.0, 'Evolved Looker Up 1_1_1': 1.0, 'Evolved Looker Up 2_2_2': 1.0, 'Exploratory Lenient Grim 2': 0.9052120592743996, 'Exploratory Tit For 3 Tats': 0.9023932426091037, 'FAWS': 0.11939937030758052, 'False Cooperator': 0.04611482591653217, 'Falsity': 1.0, 'Female / Woman': 0.9755112345367332, 'Fibonacci': 0.5428642220019821, 'Fictitious Play': 0.002053388090349076, 'Finger Print': 0.01444043321299639, 'Firm But Fair / Firm For Tat': 1.0, 'First By Davis': 1.0, 'First By Downing': 0.0, 'First By Downing, But Get Revised By Axelrod': 1.0, 'First By Feld': 0.7574257425742574, 'First By Graaskamp / Second By Mauk': 0.9951655789219241, 'First By Grofman': 1.0, 'First By Grudger / Grim Trigger': 1.0, 'First By Joss / Naive Prober': 0.8986580286904211, 'First By Named Withheld': 0.7777777777777778, 'First By Nydegger': 1.0, 'First By Shubik': 1.0, 'First By Stein & Rapoport': 0.9406267179769104, 'First By Tideman & Chieruzzi': 1.0, 'First By Tit For Tat': 1.0, 'First By Tullock': 0.9074596257369905, 'Fixed Pattern': 0.5, 'Focal Point Strategy / Schelling Point Strategy': 0.0, 'Fool Me Once': 1.0, 'Forest-RND': 1.0, 'Forest-XG': 1.0, 'Forgetable Gradual': 1.0, 'Forgetful Fool Me Once': 1.0, 'Forgetful Grudger': 1.0, 'Forgetful Majority Bot': 1.0, 'Forgiver': 1.0, 'Forgiver Benjamin M. Zagorsky & Johannes G. Reiter & Krishnendu Chatterjee & Martin A. Nowak Version / S14': 1.0, 'Forgiving': 1.0, 'Forgiving Fate': 1.0, 'Forgiving Grudger': 1.0, 'Forgiving Tit For Tat': 1.0, 'Fortress 3': 0.0, 'Fortress 4': 0.0, 'Frankl': 1.0, 'Free Rider': 0.0, 'Frequency Analyzer': 1.0, 'Freud': 1.0, 'Friendly Human': 1.0, 'GT-E / Game Theory Explorer': 1.0, 'Gambler Attariq Version': 0.5135579293344289, 'Gaslighter': 0.9489411764705883, 'Geller': 1.0, 'Geller Cooperator': 1.0, 'Geller Defector': 1.0, 'Gen 1_64_200_30_0.3_A': 0.0, 'Gen 2_64_200_30_0.3_A': 0.00454054054054054, 'Gen 3_64_200_30_0.3_A': 0.000242306760358614, 'Gen 4_64_200_30_0.3_A': 0.0, 'Gen 5_64_200_30_0.3_A': 0.49607423269093504, 'Generic Downing': 0.02377555872563005, 'Generic Downing Friendly': 0.025846471956577927, 'Generous Tit For Tat': 1.0, 'Generous Tit For Tat Axelrod Project Contributor Team Version': 1.0, 'Generous Tit For Tat Benjjo Version': 1.0, 'Generous Tit For Tat Eckhart Arnold Version 1': 1.0, 'Generous Tit For Tat Eckhart Arnold Version 2': 1.0, 'Generous Tit For Tat Less Wrong Version': 1.0, 'Genetic Algo 1': 1.0, 'Genetic Algo 2': 1.0, 'George': 0.9435367611725132, 'Go By Majority / Second By Grisell / Soft Go By Majority': 1.0, 'Go By Minority': 0.005134788189987163, 'Gold Digger': 0.0, 'Golden Mean': 1.0, 'Good': 1.0, 'GossipAgent': 0.9977077363896848, 'Graaskamp Benjjo Version': 0.9824561403508771, 'Gradual': 1.0, 'Gradual Cristal Version': 1.0, 'Gradual Killer': 0.9750561237216263, 'Grateful / S9': 0.9950580677044725, 'Greedy': 1.0, 'Grudger Alternator': 1.0, 'Grumpy': 1.0, 'HMM Player': 0.49715099715099714, 'Half Sin Square': 1.0, 'Handshake': 0.0051059484299208575, 'Hard Go By Majority': 1.0, 'Hard Prober': 0.01016002032004064, 'Hard Tit For 2 Tats': 1.0, 'Hard Tit For Tat': 1.0, 'Hesitant Q-Learner': 0.505861365953109, 'Hill Climb': 0.09914872308462694, 'Hill Climb Attariq Version': 0.9954992967651195, 'Hopeless': 0.5, 'Hot Coffee': 1.0, 'Humpty 10': 0.17323232323232324, 'Hungry': 0.9642401021711366, 'ISO': 1.0, 'Inequality / SOVIET UNION! / USSR': 0.8999743523980508, 'Inequity Averse Q-Learner': 0.5297393065046824, 'Inverse': 1.0, 'Inverse Punisher': 1.0, 'Is There Something New?': 1.0, 'Isolated Clan': 0.9512024282045295, 'Isolated Tit For Tat': 1.0, 'Japan Empire Emperor': 1.0, 'Japan Empire Soldier 1': 1.0, 'Japan Empire Soldier 10': 1.0, 'Japan Empire Soldier 2': 1.0, 'Japan Empire Soldier 3': 1.0, 'Japan Empire Soldier 4': 1.0, 'Japan Empire Soldier 5': 1.0, 'Japan Empire Soldier 6': 1.0, 'Japan Empire Soldier 7': 1.0, 'Japan Empire Soldier 8': 1.0, 'Japan Empire Soldier 9': 1.0, 'Javelin': 1.0, 'Jem': 0.8829865361077112, 'Jester': 0.8951917882225824, 'Jesus Petry': 0.8958389261744967, 'Josuah': 0.5041511594617807, 'Jung': 1.0, 'Karen': 0.9015881280916428, 'Keyman': 0.8020893789901334, 'Killer': 0.009891196834817012, 'King Of 48 Laws Of Power': 1.0, 'Knowledgeable Worse & Worse': 0.4870830198143968, 'Konflikt Blue': 0.8986418511066399, 'Konflikt Green': 1.0, 'L-BODOR / Learning Based On Derivative Of Reward': 1.0, 'LCG': 0.4236526946107784, 'Laran': 0.005275652862041678, 'Laws': 1.0, 'Learner': 1.0, 'Lenient Grim 2': 1.0, 'Lenient Grim 3': 1.0, 'Less Grim': 1.0, 'Level Punisher': 1.0, 'Liar Person': 0.8015248401377275, 'Lighty-Darky': 0.9024128686327078, 'Limited Retaliate': 1.0, 'Limited Retaliate 2': 1.0, 'Limited Retaliate 3': 1.0, 'Line': 1.0, 'Little Grudger': 1.0, 'Local Copycat': 0.586171310629515, 'Long Horse': 0.9836808703535811, 'Long Term Strategist': 1.0, 'Longterm Tit For Tat': 1.0, 'Look Ahead / Look Up': 0.009380863039399626, 'Loyal': 1.0, 'MARS / Mimicry And Relative Similarity': 1.0, 'MCMC / Markov Chain Monte Carlo': 1.0, 'MEM2': 1.0, 'MENACE': 1.0, 'MSAgent': 0.9884526558891455, 'Mafia A': 0.00496031746031746, 'Mafia B': 0.4958183990442055, 'Mafia C': 0.005125576627370579, 'Mafia D': 0.48318804483188044, 'Majapahit': 1.0, 'Male / Man': 1.0, 'Malthrin': 0.9414402532313374, 'Man In The Middle': 0.9033766233766234, 'Manipulator': 0.014698677119059285, 'Manipulity': 0.2965936739659367, 'Markov': 1.0, 'Matcher': 1.0, 'Math Constant Hunter': 1.0, 'Memory 1': 0.5200106016432547, 'Memory Decay': 1.0, 'Mensa': 0.015483870967741935, 'Meta Hunter': 0.5283714075165807, 'Meta Hunter Aggressive': 0.5005149330587023, 'Meta Majority': 1.0, 'Meta Minority': 0.0, 'Meta Mixer': 0.7892644135188867, 'Meta Winner': 0.007699711260827719, 'Meta Winner Ensemble': 0.005111167901865576, 'Michaelos': 0.8802108289410637, 'Minimax Strategy': 0.0, 'Mirror': 1.0, 'Mixed Strategy': 0.0, 'Modal Defector': 0.004694835680751174, 'Modal Tit For Tat': 1.0, "Modified Primer's Strategy": 0.009282896263634254, 'Momentum': 1.0, 'Monkey See, Monkey Do': 0.9952130205840115, 'Mosquito': 0.20280580851587496, 'MrX': 0.0, 'Mystical Logic': 0.9063382088786246, 'N Pavlov': 0.9657999531506207, 'N Tits For M Tats': 1.0, 'Naive Q-Learner': 0.6627140974967062, 'Nash': 1.0, 'Nasty Tit For Tat': 0.7989642184557438, 'Negation': 0.002457606291472106, 'Neil A': 1.0, 'Newton': 0.9969512195121951, 'Nice Meta Winner': 1.0, 'Nice Meta Winner Ensemble': 1.0, 'Nydegger Benjjo Version': 1.0, 'Observant': 1.0, 'Odd-Even Go By Majority': 1.0, 'Odysseus / Ὀδυσσεύς': 1.0, 'Omega Tit For Tat': 1.0, 'Once Bitten': 1.0, 'Opportunist': 0.005242463958060288, 'Opposite Grudger': 0.9952038369304557, 'Out For Tat': 1.0, 'PIN': 0.9948744233726294, 'PSO Gambler 1_1_1': 1.0, 'PSO Gambler 2_2_2': 1.0, 'PSO Gambler 2_2_2 Noise 05': 1.0, 'PSO Gambler Mem 1': 1.0, 'Paradoxic Grim / S21': 0.0054585152838427945, 'Parameterized Tit For Tat': 0.9528662420382166, 'Paranoid': 0.7632800412583806, 'Passive Tit For Tat': 1.0, 'Patient Cooperator': 1.0, 'Pattern Reader': 1.0, 'Patterned Adaptive Fortress': 1.0, 'Patterned Adaptive Fortress 2': 1.0, 'Pavlov / Win-Stay, Lose-Shift': 1.0, 'Pavlov D / Suspicious Pavlov': 0.0, 'Pavlov Tester': 0.3319572608062166, 'Pavolovo': 0.05403541043918142, 'Pawlow': 1.0, 'Payoff-Dominance Strategy': 1.0, 'Perceptron': 0.4401008827238335, 'Phi': 0.004602991944764097, 'Pi': 0.00481811611659841, 'Prabowo': 0.11312849162011174, 'Praedator': 0.15718157181571815, 'Predator': 0.004580852038479157, "Primer's Strategy": 0.0, 'Probable Coop': 0.7522219229733369, 'Probe & Exploit': 0.9823641563393708, 'Probe & Punish': 1.0, 'Prober': 0.010866612333604999, 'Prober 2': 0.9954149472718936, 'Prober 3': 0.004847309743092583, 'Prober 4': 0.045341192473362046, 'Prober Abdullah-arif-swe Version': 0.01001001001001001, 'Procastinater': 1.0, 'Psycho Attariq Version': 1.0, 'Pun 1': 0.9955056179775281, 'Punisher': 1.0, 'Pythagoras': 0.004997501249375313, 'Q-Learning A-Learner': 1.0, 'REINFORCE': 0.4175013178703216, 'RLAgent': 0.8481012658227848, 'RLGossipAgent': 0.5, 'RNN & RTRL': 0.6627646326276463, 'Racister': 0.5197505197505198, 'Rack Block Shooter': 0.9724517906336089, 'Raider': 0.0, 'Random': 0.5068211068211068, 'Random Hunter': 1.0, 'Random Tit For Tat': 0.7487179487179487, "Random's Exploiter 1": 1.0, "Random's Exploiter 2": 1.0, "Random's Exploiter 3": 1.0, "Random's Exploiter 4": 1.0, 'Reactive Player': 0.5823074720870312, 'Rememberable Grudger': 1.0, 'Remorseful Prober': 0.8897980056251598, 'Resurrection': 1.0, 'Retaliate': 1.0, 'Retaliate 2': 1.0, 'Retaliate 3': 1.0, 'Rice Field': 0.9184466019417475, 'Ripoff': 0.49876053544868615, 'Risk-Dominance Strategy': 0.0, 'Risky Q-Learner': 0.5262271414821944, 'Rogers & Maslow': 1.0, 'Rover': 0.0, 'Rwallace': 0.8888354186717998, 'S11': 0.49834983498349833, 'S12 / Suspicious Forgiver': 0.9949354266903013, 'S18': 0.5009508285791905, 'S19': 0.5015502027188171, 'S2': 0.0, 'S23': 0.5014749262536873, 'S24': 0.004748338081671415, 'S6': 0.4987951807228916, 'S7': 0.49813486370157817, 'SURPRISE ATTACK!': 0.5688591983556013, 'Sample / Second By Tit For 2 Tats': 1.0, 'Sarsa Q-Learner': 0.5178617322025186, 'Schwoba Sen Gnitz': 1.0, 'Scorched Earth': 1.0, 'Score Equalizer': 1.0, 'Second By Almy': 0.52734375, 'Second By Ambuehl & Hickey': 1.0, 'Second By Anderson': 0.7671717171717172, 'Second By Appold': 1.0, 'Second By Batell': 1.0, 'Second By Black': 1.0, 'Second By Borufsen': 1.0, 'Second By Cave': 1.0, 'Second By Champion': 1.0, 'Second By Colbert': 0.9949647532729103, 'Second By Dawes & Batell': 1.0, 'Second By Eatherley': 1.0, 'Second By Falk & Langsted': 1.0, 'Second By Feld': 0.17840497151734852, 'Second By Friedland': 0.00999000999000999, 'Second By Friedland, But Get Revised By Axelrod': 0.010113780025284451, "Second By GRASR's Author": 0.9300237655135991, 'Second By George Hufford': 0.07643312101910828, 'Second By Getzler': 1.0, 'Second By Gladstein / Tester': 0.5012224938875306, 'Second By Graaskamp & Katzen': 1.0, 'Second By Grofman': 1.0, 'Second By Hall': 0.5012493753123438, 'Second By Harrington': 0.7513416815742398, 'Second By Hollander': 0.9194424757854949, 'Second By Hotz': 0.08423529411764706, 'Second By Jones': 1.0, 'Second By Joss': 0.9116822429906543, 'Second By Kluepfel': 1.0, 'Second By Lefevre': 1.0, 'Second By Leyland': 0.2717258261933905, 'Second By Leyvraz': 1.0, 'Second By Mcgurrin': 0.8738127544097694, 'Second By Mikkelson': 1.0, 'Second By Newman': 1.0, 'Second By Nussbacher': 1.0, 'Second By Nydegger': 1.0, 'Second By Pebly': 1.0, 'Second By Pinkley': 1.0, 'Second By Quayle': 1.0, 'Second By Rabbie': 0.055645722235103175, 'Second By Richard Hufford': 0.9947187747557433, 'Second By Robert Adams': 1.0, 'Second By Robertson': 0.9053446553446554, 'Second By Rowsam': 1.0, 'Second By Schurmann': 1.0, 'Second By Smith': 0.9545923632610939, 'Second By Smoody': 0.8954703832752613, 'Second By Snodgrass': 0.3555997020114229, 'Second By Tideman & Chieruzzi': 1.0, 'Second By Tranquilizer': 0.8295994568906992, 'Second By Weiderman': 1.0, 'Second By Weiner': 1.0, 'Second By White': 1.0, 'Second By Wiliam Adams': 1.0, 'Second By Yamachi': 1.0, 'Second By Zimmerman': 1.0, 'Second Chance': 0.9258966244725738, 'Second Chance Attariq Version': 1.0, 'Self Steem': 0.696124763705104, 'Self-Love A-Learner': 1.0, 'Short Mem': 1.0, 'Signaling Cheater': 0.010111223458038422, 'Simple ANN': 0.532046332046332, 'Simple Exploiter': 0.004598758335249482, 'Simple Identity ChecK': 0.30319148936170215, 'Simpleton': 0.9951016409502816, 'Skinner': 1.0, 'Slanger': 1.0, 'Slow Tit For Tat': 1.0, 'Smart': 1.0, 'Smart Cat': 1.0, 'Smart Generous Tit For Tat': 1.0, 'Smart Go By Majority': 1.0, 'Smart Snake': 1.0, 'Smart-Clever': 1.0, 'Smarter Tat For Tit': 0.994386752736458, 'Smooth Grudger': 1.0, 'Smooth Tit For Tat': 1.0, 'Sneaky Tit For Tat': 0.0102880658436214, 'Social Average': 0.9006052141527002, 'Social Engineering': 0.9860917941585535, 'Socio': 1.0, 'Soft Grim': 1.0, 'Soft Grudger': 1.0, 'Solution B1': 0.9902818270165209, 'Solution B5': 0.9947685064085796, 'Sort': 1.0, 'Spectrum Attacker': 0.7786615469007131, 'Spiteful CC': 1.0, 'Spiteful Tit For Tat': 1.0, 'Split Brain Syndrome': 1.0, 'Springter': 1.0, 'Stabilize': 1.0, 'Stalker': 1.0, 'Star S': 0.015376729882111738, 'Star SL': 0.9903147699757869, 'Star SN': 0.015071590052750565, 'Star Slave 1': 0.015544041450777202, 'Star Slave 10': 0.014566642388929352, 'Star Slave 2': 0.015491866769945779, 'Star Slave 3': 0.01509433962264151, 'Star Slave 4': 0.014906832298136646, 'Star Slave 5': 0.015113350125944584, 'Star Slave 6': 0.014862521674510775, 'Star Slave 7': 0.014619883040935672, 'Star Slave 8': 0.016611295681063124, 'Star Slave 9': 0.014673514306676448, 'Statistical Strategy': 0.16997238262616118, 'Stefan Pennartz': 1.0, 'Stochastic Cooperator': 0.8022766641920317, 'Stochastic WSLS': 0.5384615384615384, 'Stoic Mirror': 1.0, 'Stoicalist': 1.0, 'Strategic Commitment': 0.0, 'Strategist': 0.004847309743092583, 'Stupidiot': 0.5018597236981934, 'Sugar Strategy / Treasure Hunt': 0.9952107279693486, 'Sumobox Fighter': 1.0, 'Sun Tzu Bot': 1.0, 'Suspicious Alternator': 0.4989373007438895, 'Suspicious Tit For Tat': 0.9950433705080545, 'Sweet Heart': 0.9408298551436288, 'Syifa': 0.046945986875315496, 'TF 1': 0.503199810381607, 'TF 2': 0.600241837968561, 'TF 3': 1.0, 'TF2T & 2TFT': 1.0, 'TOM Level 1 / Theory Of Mind Level 1': 0.005016302984700276, 'TOM Level 2 / Theory Of Mind Level 2': 0.6987012987012987, 'TOM Level 3 / Theory Of Mind Level 3': 1.0, 'TPMS / Tree Predictive Matrix Strategy': 1.0, 'Tages': 0.20609113808106252, 'Tau': 0.004964010920824026, 'Teacher Pee While Standing, Student Pee While Running': 1.0, 'Tester Eckhart Arnold Version': 0.49368029739776953, 'The Calculated Mirror With A Fuse': 0.9944506104328524, 'The Cave Breaker': 0.8027644869750132, 'The Nasty A': 0.8625711035267349, 'Theseus': 1.0, 'Three Times Out': 1.0, 'Threshold Punisher': 1.0, 'Thresholded Trust Bot': 1.0, 'Thue Morse': 0.5, 'Thue Morse Inverse': 0.5043295108822841, 'Thumper': 1.0, 'Tiny Brain': 0.97458071278826, 'Tip For Tap': 1.0, 'Tit For 3 Tats': 1.0, 'Tit For Increasing Tat': 1.0, 'Tit For Tat But Not Too Forgiving': 1.0, 'Tit For Tat Killer': 0.9950787401574803, 'Toxic Mirror': 0.8600345593680573, 'Trader': 1.0, 'Tranquilizer Eckhart Arnold Version': 0.004642525533890436, 'Tri-Brain': 1.0, 'Tricky Cooperator': 0.015560165975103735, 'Tricky Defector': 0.0, 'Tricky Level Punisher': 1.0, 'Trust Ledger': 1.0, 'Trust Score Sigmoid': 0.9945510542525468, 'Trusty': 1.0, 'Turan': 0.06986899563318777, 'Tweedledee': 1.0, 'Tweedledum': 1.0, 'Tweetypie': 0.0, 'Unpredictable Tit For Tat': 0.9015990159901599, 'Uranium': 1.0, 'Usually Cooperates': 1.0, 'Usually Defects': 0.0, 'Vandal': 0.7514595496246872, 'Vengeful': 1.0, 'Vengeful Cheater': 0.8755576426391172, 'Verity': 1.0, 'Very Bad': 1.0, 'WHAgent': 0.9977151561309977, 'Whale': 1.0, 'William': 1.0, 'Willing': 0.9980824544582934, 'Win-Shift, Lose-Stay': 0.498468606431853, 'Winner 12': 1.0, 'Winner 21': 1.0, 'Worse & Worse / Worse & Worse 1': 0.898969590349334, 'Worse & Worse 2': 0.32400379506641364, 'Worse & Worse 3': 1.0, 'YOUR CATEGORY': 1.0, 'Your Choice Or My Choice': 1.0, 'ZD-2026 / Zero Determinant 2026': 1.0, 'ZD-G / Zero Determinant Generous': 1.0, 'ZD-GTFT / Zero Determinant Generous Tit For Tat': 1.0, 'ZD-M2 / Zero Determinant Memory 2': 0.8750607681088964, 'ZD-MD / Zero Determinant Middle': 0.7117726657645467, 'ZD-MS / Zero Determinant Mischief': 0.34346959122632104, 'ZD-PW2026 / Zero Determinant Psychological War 2026': 1.0, 'ZD-Q / Zero Determinant Equalizer': 0.6806647204132046, 'ZD-S / Zero Determinant Set': 0.6603128054740958, 'ZD-X / Zero Determinant Extortion': 0.6355685131195336, 'ZD-X A. J. Stewart & J. B. Plotkin Version / Zero Determinant Extortion A. J. Stewart & J. B. Plotkin Version': 0.7413406429105407, 'ZD-X Lars Roemheld Version / Zero Determinant Extortion Lars Roemheld Version': 0.44107475392391593, 'ZD-X Marc Harper Version / Zero Determinant Extortion Marc Harper Version': 0.5630731102850062, 'ZD-X Steven Kuhn Version / Zero Determinant Extortion Steven Kuhn Version': 0.7466334164588528, 'Zero Day': 0.8733982573039467, 'Zero Input Apery': 0.7759011847743887, 'Zero Input Champernowne': 0.5593017694882831, 'Zero Input Euler': 0.8645336963921034, 'Zero Input Euler-Mascheroni': 0.6860186418109188, 'Zero Input Phi': 0.8123138033763655, 'Zero Input Pi': 0.8691112808759868, 'Zero Input Pythagoras': 0.7969828688315009, 'Zero Input Tau': 0.9259149940968123, 'ZeroResp': 1.0, 'ZeroResp V5.3': 0.898761835396941}

strategies_many_name = []
for i in range(len(strategies)):
    if '/' in strategies[i]:
        strategies_many_name.append(strategies[i])

strategies_inf_loop_maker = ['Geller', 'Geller Cooperator', 'Geller Defector', 'Attariq\'s Cheating Matching Pennies Strategy', 'Adaptable Game type Strategy']

population_type = tuple([strategies_category[i] for i in strategies])

default_memory_looked = 10
tournament_avg_round = 200
tournament_avg_last_round = tournament_avg_round - 1
tournament_round_randomness = 75
axelrod_s_address = "[axe@umich.edu]"

if game_payoff_matrix[0] == game_payoff_matrix[1]:
    if (game_payoff_matrix[0]['DC'] > game_payoff_matrix[0]['CC'] > game_payoff_matrix[0]['DD'] > game_payoff_matrix[0]['CD']):
        game_type = 'Prisoner\'s Dilemma'
    elif (game_payoff_matrix[0]['DC'] > game_payoff_matrix[0]['CC'] > game_payoff_matrix[0]['CD'] > game_payoff_matrix[0]['DD']):
        game_type = 'Game Of Chicken / Hawk-Dove'
    elif (game_payoff_matrix[0]['CC'] > game_payoff_matrix[0]['DC'] == game_payoff_matrix[0]['DD'] > game_payoff_matrix[0]['CD']):
        game_type = 'Stag Hunt'
    else:
        game_type = 'Unknown / Other Game'
else:
    if any((game_payoff_matrix[i]['DD'] > game_payoff_matrix[i]['CC'] > game_payoff_matrix[i]['DC'] == game_payoff_matrix[i]['CD']) and (game_payoff_matrix[1 - i]['CC'] > game_payoff_matrix[1 - i]['DD'] > game_payoff_matrix[1 - i]['DC'] == game_payoff_matrix[1 - i]['CD']) for i in range(2)):
        game_type = "Battle Of Sexes"
    elif any((game_payoff_matrix[i]['CC'] == game_payoff_matrix[i]['DD'] > game_payoff_matrix[i]['DC'] == game_payoff_matrix[i]['CD']) and (game_payoff_matrix[1 - i]['DC'] == game_payoff_matrix[1 - i]['CD'] > game_payoff_matrix[1 - i]['CC'] == game_payoff_matrix[1 - i]['DD']) for i in range(2)):
        game_type = 'Matching Pennies'
    else:
        game_type = 'Unknown / Other Game'

if __name__ == "__main__":
    memory = []
    game_score = [0, 0]
    strategy_chat = ['', '']
    strategy_next_chat = [[''], ['']]
    strat_memory = [reset_internal_memory(0), reset_internal_memory(1)]
    guessing = False
    the_play = input('play game, versus, strategy list, check comman list, or tournament? ')
    if the_play != 'tournament':
        if the_play != 'versus':
            if the_play != 'strategy creator':
                if the_play != 'strategy category':
                    if the_play != 'check current index':
                        if the_play != 'strategy list':
                            if the_play != 'number of strategies':
                                if the_play != 'NICE:NASTY':
                                    if the_play != 'check comman list':
                                        if the_play != 'sort':
                                            if the_play != 'guess random strategy':
                                                if the_play != 'categorize & coop prob':
                                                    if the_play != 'search':
                                                        if the_play != 'strategies that not from this tournament':
                                                            if the_play != 'strategies that from this tournament':
                                                                if the_play != 'game type':
                                                                    input_strategy = input('enemy strategy: ')
                                                                    if input_strategy.lower() != 'player':
                                                                        the_strategy = many_name_strategies_function(input_strategy)
                                                                        print(f"\033[{1};1H\033[K", end="")
                                                                        print(f"\033[{2};1H\033[K", end="")
                                                                        the_play = 'play game'
                                                                    else:
                                                                        the_play = 'play with friends'
                                                                else:
                                                                    print(game_type)
                                                            else:
                                                                print([i for i in strategies if any(j in strategies_creator[i] for j in ('M. A. M. Attariq', 'Gemini', 'Chat-GPT', 'Claude', 'Dola', 'DeepSeek', 'Perplexity', 'Copilot', 'Meta'))])
                                                        else:
                                                            print([i for i in strategies if not any(j in strategies_creator[i] for j in ('M. A. M. Attariq', 'Gemini', 'Chat-GPT', 'Claude', 'Dola', 'DeepSeek', 'Perplexity', 'Copilot', 'Meta'))])
                                                    else:
                                                        searching = input('search what? ')
                                                        if '[CREATED BY]' not in searching:
                                                            print([i for i in strategies if searching in i])
                                                        else:
                                                            print([i for i in strategies if searching.replace('[CREATED BY]', '') in strategies_creator[i]])
                                            else:
                                                the_strategy = rnd.choice(strategies)
                                                print(f"\033[{1};1H\033[K", end="")
                                                print(f"\033[{2};1H\033[K", end="")
                                                the_play = 'play game'
                                                guessing = True
                                        else:
                                            print(f'strategies = {sorted(strategies)}')
                                            print(f'strategies_creator = {dict(sorted(strategies_creator.items()))}')
                                            print(f'strategies_category = {dict(sorted(strategies_category.items()))}')
                                            print(f'strategies_coop_prob = {dict(sorted(strategies_coop_prob.items()))}')
                                    else:
                                        print('play game'); print('tournament'); print('versus')
                                        print('strategy creator'); print('strategy category')
                                        print('check current index'); print('strategy list')
                                        print('number of strategies'); print('NICE:NASTY')
                                        print('check comman list'); print('sort'); print('guess random strategy')
                                        print('categorize & coop prob'); print('search')
                                        print('strategies that not from this tournament')
                                        print('strategies that from this tournament')
                                        print('game type')
                                else:
                                    NICE_sum = sum(1 for i in range(len(strategies)) if strategies_category[strategies[i]] == 'NICE')
                                    NASTY_sum = len(strategies) - NICE_sum
                                    print(f"{NICE_sum}:{NASTY_sum}")
                            else:
                                print(len(strategies))
                        else:
                            print(f'strategy list: {strategies[:]}')
                    else:
                        print(len(strat_memory[1]) - 1)
                else:
                    print(strategies_category.get(many_name_strategies_function(input('strategy code name? '))))
            else:
                print(strategies_creator.get(many_name_strategies_function(input('strategy code name? '))))
        else:
            strategy1 = many_name_strategies_function(input('first strategy? '))
            strategy2 = many_name_strategies_function(input('second strategy? '))
        if the_play == 'categorize & coop prob':
            strategy_score = [0] * len(strategies)
            strategies_category = {i: 'NICE' for i in strategies}
            strategies_coop_total = {i: 0 for i in strategies}
            total_game_round = {i: 0 for i in strategies}
            tournament_try = 20
            pairs = [(i, strategies.index('Always Cooperate / Cooperator / Dove / Saint')) for i in range(len(strategies))]
            tasks = [(i, j, False, 0) for i, j in pairs for _ in range(tournament_try)]
            print(f'processing categorize with {len(strategies)} strategies across {mp.cpu_count()} cores')
            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = pool.map(run_clasification, tasks)
            for i, j, ct1, g in results:
                if ct1 < g:
                    strategies_category[strategies[i]] = 'NASTY'
                strategies_coop_total[strategies[i]] += ct1
                total_game_round[strategies[i]] += g
            strategies_coop_prob = {i: ((strategies_coop_total[i] / total_game_round[i]) if total_game_round[i] > 0 else 0.5) for i in strategies}
            print(f'strategies_category = {strategies_category}')
            print(f'strategies_coop_prob = {strategies_coop_prob}')
        if the_play in ('play game', 'versus'):
            game_round = 0
            tournament_round = tournament_avg_round + rnd.randint(-tournament_round_randomness, tournament_round_randomness)
            while True:
                strategy_next_chat = [[''], ['']]
                if the_play != 'versus':
                    sh_st = timeout_input_instant(0.2)
                    if sh_st in ('q', 'w', 'e', 's', 'Q', 'W', 'E', 'S'):
                        if sh_st.lower() == 'q':
                            player_choice = 'C'
                        elif sh_st.lower() == 'w':
                            player_choice = 'D'
                        elif sh_st.lower() == 's':
                            player_choice = '[OUT]'
                        else:
                            player_choice = rnd_prob(0.5)
                        enemy_choice = call_strategy(the_strategy, memory, 0, 'Player', strat_memory[1], strat_immortal_memory[1], strategy_chat, strategy_next_chat[1], game_score, population_type)
                        if enemy_choice not in ('C', 'D', '[OUT]'):
                            raise ValueError(f'ERROR {the_strategy}')
                        if enemy_choice == '[OUT]' or player_choice == '[OUT]':
                            break
                        memory.append((player_choice, enemy_choice))
                        if player_choice == enemy_choice:
                            update_score(game_payoff_matrix[0]['CC'], game_payoff_matrix[1]['CC']) if player_choice == 'C' else update_score(game_payoff_matrix[0]['DD'], game_payoff_matrix[1]['DD'])
                        else:
                            update_score(game_payoff_matrix[0]['DC'], game_payoff_matrix[1]['CD']) if player_choice == 'D' else update_score(game_payoff_matrix[0]['CD'], game_payoff_matrix[1]['DC'])
                        game_round += 1
                else:
                    strategy1_choice = call_strategy(strategy1, memory, 1, strategy2, strat_memory[0], strat_immortal_memory[0], strategy_chat, strategy_next_chat[0], game_score, population_type)
                    strategy2_choice = call_strategy(strategy2, memory, 0, strategy1, strat_memory[1], strat_immortal_memory[1], strategy_chat, strategy_next_chat[1], game_score, population_type)
                    if (strategy1_choice not in ('C', 'D', '[OUT]')) or (strategy2_choice not in ('C', 'D', '[OUT]')):
                        ERROR = []
                        if strategy1_choice not in ('C', 'D', '[OUT]'):
                            ERROR.append(strategy1)
                        if strategy2_choice not in ('C', 'D', '[OUT]'):
                            ERROR.append(strategy2)
                        raise NameError(f'ERROR {ERROR}')
                    if '[OUT]' in (strategy1_choice, strategy2_choice):
                        break
                    memory.append((strategy1_choice, strategy2_choice))
                    if strategy1_choice == strategy2_choice:
                        update_score(game_payoff_matrix[0]['CC'], game_payoff_matrix[1]['CC']) if strategy1_choice == 'C' else update_score(game_payoff_matrix[0]['DD'], game_payoff_matrix[1]['DD'])
                    else:
                        update_score(game_payoff_matrix[0]['DC'], game_payoff_matrix[1]['CD']) if strategy1_choice == 'D' else update_score(game_payoff_matrix[0]['CD'], game_payoff_matrix[1]['DC'])
                    game_round += 1
                    time.sleep(0.05)
                strategy_chat[0] = strategy_next_chat[0][0]
                strategy_chat[1] = strategy_next_chat[1][0]
                print("\033[H", end="")
                for i in range(2):
                    color = ['', '', '', '', '']
                    bad_space = f"\033[38;2;{255};{128};{0}m{'░░'}\033[0m"
                    good_space = f"\033[38;2;{150};{255};{0}m{'░░'}\033[0m"
                    for j in range(len(memory) - min(default_memory_looked, game_round), len(memory)):
                        color[0] += good_space + f"\033[38;2;{0};{255};{0}m{'░░░░░░'}\033[0m" + good_space if memory[j][i] == 'C' else f"\033[38;2;{255};{0};{0}m{'░░░░░░'}\033[0m" + (bad_space * 2)
                        color[1] += f"\033[38;2;{0};{255};{0}m{'░░'}\033[0m" + (good_space * 4) if memory[j][i] == 'C' else f"\033[38;2;{255};{0};{0}m{'░░'}\033[0m" + (bad_space * 2) + f"\033[38;2;{255};{0};{0}m{'░░'}\033[0m" + bad_space
                        color[2] += f"\033[38;2;{0};{255};{0}m{'░░'}\033[0m" + (good_space * 4) if memory[j][i] == 'C' else f"\033[38;2;{255};{0};{0}m{'░░'}\033[0m" + (bad_space * 2) + f"\033[38;2;{255};{0};{0}m{'░░'}\033[0m" + bad_space
                        color[3] += f"\033[38;2;{0};{255};{0}m{'░░'}\033[0m" + (good_space * 4) if memory[j][i] == 'C' else f"\033[38;2;{255};{0};{0}m{'░░'}\033[0m" + (bad_space * 2) + f"\033[38;2;{255};{0};{0}m{'░░'}\033[0m" + bad_space
                        color[4] += good_space + f"\033[38;2;{0};{255};{0}m{'░░░░░░'}\033[0m" + good_space if memory[j][i] == 'C' else f"\033[38;2;{255};{0};{0}m{'░░░░░░'}\033[0m" + (bad_space * 2)
                    print(color[0]); print(color[1]); print(color[2]); print(color[3]); print(color[4])
                    if i == 0:
                        print(f"\033[38;2;{255};{255};{0}m{'░░░░░░░░░░'}\033[0m" * default_memory_looked)
                baris_pesan = 12
                print(f"\033[{baris_pesan};1H\033[K", end="")
                if the_play != 'versus':
                    if not guessing:
                        print(f'player: {game_score[0]}, NETRAL, \"\" | {the_strategy}: {game_score[1]}, {strategies_category.get(the_strategy)}, \"{strategy_chat[1]}\" | game round: {game_round} | memory: {strat_memory[1][423][0]}')
                    else:
                        print(f'player: {game_score[0]}, NETRAL, \"\" | ???: {game_score[1]}, NETRAL, \"{strategy_chat[1]}\" | game round: {game_round}')
                else:
                    print(f'{strategy1}: {game_score[0]}, {strategies_category.get(strategy1)}, \"{strategy_chat[0]}\" | {strategy2}: {game_score[1]}, {strategies_category.get(strategy2)}, \"{strategy_chat[1]}\" | game round: {game_round}')
                if game_round >= tournament_round:
                    if guessing:
                        print(memory)
                        the_guess = many_name_strategies_function(input('What Is The Identity Of This Strategy? '))
                        if the_guess == the_strategy:
                            print('YOU RIGHT!')
                        else:
                            print(f'YOU WRONG!\nTHE ANSWER IS {the_strategy}')
                    break
            if not guessing:
                print(memory)
        elif the_play == 'play with friends':
            game_round = 0
            tournament_round = tournament_avg_round + rnd.randint(-tournament_round_randomness, tournament_round_randomness)
            players_turn = 1
            while True:
                sh_st = timeout_input_instant(0.2)
                if sh_st in ('q', 'w', 'e', 's', 'Q', 'W', 'E', 'S'):
                    strategy_next_chat = [[''], ['']]
                    if sh_st.lower() == 'q':
                        if players_turn == 1:
                            player1_choice = 'C'
                        else:
                            player2_choice = 'C'
                    elif sh_st.lower() == 'w':
                        if players_turn == 1:
                            player1_choice = 'D'
                        else:
                            player2_choice = 'D'
                    else:
                        if players_turn == 1:
                            player1_choice = rnd_prob(0.5)
                        else:
                            player2_choice = rnd_prob(0.5)
                    if players_turn == 2:
                        memory.append((player1_choice, player2_choice))
                        if player1_choice == player2_choice:
                            update_score(game_payoff_matrix[0]['CC'], game_payoff_matrix[1]['CC']) if player1_choice == 'C' else update_score(game_payoff_matrix[0]['DD'], game_payoff_matrix[1]['DD'])
                        else:
                            update_score(game_payoff_matrix[0]['DC'], game_payoff_matrix[1]['CD']) if player1_choice == 'D' else update_score(game_payoff_matrix[0]['CD'], game_payoff_matrix[1]['DC'])
                        game_round += 1
                    players_turn += 1
                    players_turn = ((players_turn - 1) % 2) + 1
                print("\033[H", end="")
                for i in range(2):
                    color = ['', '', '', '', '']
                    bad_space = f"\033[38;2;{255};{128};{0}m{'░░'}\033[0m"
                    good_space = f"\033[38;2;{150};{255};{0}m{'░░'}\033[0m"
                    print(f"\033[{(i * 5) + 5 + i};1H\033[K", end="")
                    print(f"\033[{(i * 5) + 4 + i};1H\033[K", end="")
                    print(f"\033[{(i * 5) + 3 + i};1H\033[K", end="")
                    print(f"\033[{(i * 5) + 2 + i};1H\033[K", end="")
                    print(f"\033[{(i * 5) + 1 + i};1H\033[K", end="")
                    for j in range(len(memory) - min(default_memory_looked, game_round), len(memory)):
                        color[0] += good_space + f"\033[38;2;{0};{255};{0}m{'░░░░░░'}\033[0m" + good_space if memory[j][i] == 'C' else f"\033[38;2;{255};{0};{0}m{'░░░░░░'}\033[0m" + (bad_space * 2)
                        color[1] += f"\033[38;2;{0};{255};{0}m{'░░'}\033[0m" + (good_space * 4) if memory[j][i] == 'C' else f"\033[38;2;{255};{0};{0}m{'░░'}\033[0m" + (bad_space * 2) + f"\033[38;2;{255};{0};{0}m{'░░'}\033[0m" + bad_space
                        color[2] += f"\033[38;2;{0};{255};{0}m{'░░'}\033[0m" + (good_space * 4) if memory[j][i] == 'C' else f"\033[38;2;{255};{0};{0}m{'░░'}\033[0m" + (bad_space * 2) + f"\033[38;2;{255};{0};{0}m{'░░'}\033[0m" + bad_space
                        color[3] += f"\033[38;2;{0};{255};{0}m{'░░'}\033[0m" + (good_space * 4) if memory[j][i] == 'C' else f"\033[38;2;{255};{0};{0}m{'░░'}\033[0m" + (bad_space * 2) + f"\033[38;2;{255};{0};{0}m{'░░'}\033[0m" + bad_space
                        color[4] += good_space + f"\033[38;2;{0};{255};{0}m{'░░░░░░'}\033[0m" + good_space if memory[j][i] == 'C' else f"\033[38;2;{255};{0};{0}m{'░░░░░░'}\033[0m" + (bad_space * 2)
                    if i == (players_turn - 1):
                        color[0] += f"\033[38;2;{255};{255};{0}m{'░░░░░░░░░░'}\033[0m"
                        color[1] += f"\033[38;2;{255};{255};{0}m{'░░░░░░░░░░'}\033[0m"
                        color[2] += f"\033[38;2;{255};{255};{0}m{'░░░░░░░░░░'}\033[0m"
                        color[3] += f"\033[38;2;{255};{255};{0}m{'░░░░░░░░░░'}\033[0m"
                        color[4] += f"\033[38;2;{255};{255};{0}m{'░░░░░░░░░░'}\033[0m"
                    print(color[0]); print(color[1]); print(color[2]); print(color[3]); print(color[4])
                    if i == 0:
                        print(f"\033[38;2;{255};{255};{0}m{'░░░░░░░░░░'}\033[0m" * default_memory_looked)
                baris_pesan = 12
                print(f"\033[{baris_pesan};1H\033[K", end="")
                print(f'player 1: {game_score[0]}, NETRAL, \"\" | player 2: {game_score[1]}, NETRAL, \"\" | game round: {game_round} | turn: player {players_turn}')
                if game_round >= tournament_round:
                    print(memory)
                    break
    else:
        with_noise = input('with noise? ')
        with_noise_flag = with_noise in ('yes', 'Y', 'True', '+', 'noise')
        noise_prob = max(min(float(input('noise value? ')), 1), 0) if with_noise_flag else 0.0
        tournament_try = 5
        strategy_score = [0] * len(strategies)
        strategy_win = [0] * len(strategies)
        strategy_lose = [0] * len(strategies)
        strategy_draw = [0] * len(strategies)
        pairs = [(i, j) for i in range(len(strategies)) for j in range(i, len(strategies))]
        tasks = [(i, j, with_noise_flag, noise_prob) for i, j in pairs for _ in range(tournament_try)]
        print(f'processing tournament with {len(strategies)} strategies across {mp.cpu_count()} cores')
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(run_match, tasks, chunksize=2000)
        for i, j, s1, s2 in results:
            strategy_score[i] += s1
            strategy_score[j] += s2
            if s1 > s2:
                strategy_win[i] += 1
                strategy_lose[j] += 1
            elif s2 > s1:
                strategy_lose[i] += 1
                strategy_win[j] += 1
            else:
                strategy_draw[i] += 1
                strategy_draw[j] += 1
        time_divider = tournament_try
        tournament_divider = len(strategies) * time_divider
        gabung = list(zip(strategy_score, strategies, strategy_win, strategy_draw, strategy_lose))
        sorted_result = sorted(gabung, key=lambda x: x[0], reverse=True)
        print(f'Num | Name | Score | Category | Wins | Draws | Losses | Creator')
        for p in range(len(strategies)):
            score_p, name_p, win_p, draw_p, lose_p = sorted_result[p]
            result_color = 255 if strategies_category[name_p] == 'NICE' else 0
            print(f'\033[38;2;{255 - result_color};{result_color};{0}m{f'{p+1} | {name_p} | {round(score_p/tournament_divider,1)} | {strategies_category[name_p]} | {round(win_p/time_divider,1)} | {round(draw_p/time_divider,1)} | {round(lose_p/time_divider,1)} | {strategies_creator[name_p]}'}\033[0m')
        plot_tournament_results(sorted_result, strategies_category, strategies_creator,
                                 tournament_divider, time_divider, top_n=30)

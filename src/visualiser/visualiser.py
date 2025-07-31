#!/usr/bin/env python3

USE_DARK_THEME = False

if USE_DARK_THEME:
    COLOR_BG = '#222'
    COLOR_PAPER = '#222'
    COLOR_TEXT = 'white'
    COLOR_LABEL = 'white'
    COLOR_CIRCLE = '#444'
    COLOR_PRE_BG = '#333'
    PLOTLY_TEMPLATE = 'plotly_dark'
else:
    COLOR_BG = 'white'
    COLOR_PAPER = 'white'
    COLOR_TEXT = 'black'
    COLOR_LABEL = 'black'
    COLOR_CIRCLE = '#bbb'
    COLOR_PRE_BG = '#f0f0f0'
    PLOTLY_TEMPLATE = 'plotly_white'


import json
from collections import deque
import threading
import queue

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import zmq



# ZMQ address
ZMQ_ADDRESS = "ipc:///tmp/zmq-sim.sock"

# Queues for communication between Dash and ZMQ thread
request_queue = queue.Queue()
response_queue = queue.Queue()

def zmq_worker():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(ZMQ_ADDRESS)
    while True:
        req = request_queue.get()
        if req is None:
            break  # Shutdown signal
        try:
            socket.send_json(req)
            reply = socket.recv()
            response_queue.put(reply)
        except Exception as e:
            response_queue.put(e)

# Start the worker thread
zmq_thread = threading.Thread(target=zmq_worker, daemon=True)
zmq_thread.start()

# Store neuron history for time series plot
neuron_data = deque(maxlen=1000)  # Store last 1000 time steps


# Request the current state from the simulator using the background thread
def request_state():
    try:
        request_queue.put({"type": "get_state"})
        reply = response_queue.get(timeout=2)  # Wait for up to 2 seconds
        if isinstance(reply, Exception):
            print(f"Error requesting state from simulator: {reply}")
            return None
        data = json.loads(reply)
        return data
    except queue.Empty:
        print("Timeout waiting for simulator response")
        return None
import atexit

def shutdown_zmq_thread():
    request_queue.put(None)
    zmq_thread.join(timeout=1)

atexit.register(shutdown_zmq_thread)

def create_ring_plot(neurons):
    """Create a circular plot showing neuron activations."""
    if neurons is None:
        return go.Figure()

    n_neurons = len(neurons)

    # Calculate angles for neurons around the circle
    angles = np.linspace(0, 2*np.pi, n_neurons, endpoint=False)

    # Add the first point again to close the circle
    angles_closed = np.concatenate([angles, [angles[0]]])
    neurons_closed = neurons + [neurons[0]]

    # Create positions for the neurons
    radius = 1.0
    x_pos = radius * np.cos(angles)
    y_pos = radius * np.sin(angles)

    # Normalize neuron values for color mapping
    neuron_values = np.array(neurons)
    vmin, vmax = neuron_values.min(), neuron_values.max()
    if vmax - vmin > 0:
        normalized_values = (neuron_values - vmin) / (vmax - vmin)
    else:
        normalized_values = np.zeros_like(neuron_values)

    fig = go.Figure()

    # Add background circle
    circle_angles = np.linspace(0, 2*np.pi, 100)
    circle_x = radius * np.cos(circle_angles)
    circle_y = radius * np.sin(circle_angles)

    fig.add_trace(go.Scatter(
        x=circle_x, y=circle_y,
        mode='lines',
        line=dict(color=COLOR_CIRCLE, width=2),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Neuronal unit scatter plot
    fig.add_trace(go.Scatter(
        x=x_pos, y=y_pos,
        mode='markers',
        marker=dict(
            size=20,
            color=normalized_values,
            colorscale='viridis',
            cmin=0, cmax=1,
            colorbar=dict(title="", x=1.02),
            line=dict(width=2, color='white')
        ),
        text=[f'Neuron {i}: {val:.4f}' for i, val in enumerate(neurons)],
        hovertemplate='<b>%{text}</b><extra></extra>',
        showlegend=False
    ))

    # Add neuron labels
    label_radius = 1.15
    label_x = label_radius * np.cos(angles)
    label_y = label_radius * np.sin(angles)

    fig.add_trace(go.Scatter(
        x=label_x, y=label_y,
        mode='text',
        text=[str(i) for i in range(n_neurons)],
        textfont=dict(size=12, color=COLOR_LABEL),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Update layout
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(
            text="",
            x=0.5,
            font=dict(size=20, color=COLOR_TEXT)
        ),
        xaxis=dict(
            range=[-1.5, 1.5],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor="y",
            scaleratio=1,
            color=COLOR_TEXT
        ),
        yaxis=dict(
            range=[-1.5, 1.5],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            color=COLOR_TEXT
        ),
        plot_bgcolor=COLOR_BG,
        paper_bgcolor=COLOR_PAPER,
        width=500,
        height=500,
        margin=dict(l=50, r=100, t=50, b=50)
    )

    return fig

def create_time_series_plot(neuron_history):
    """Create a time series plot of neuron activations."""
    if not neuron_history:
        return go.Figure()

    history_array = np.array(list(neuron_history))
    n_steps, n_neurons = history_array.shape

    fig = go.Figure()

    colors = px.colors.sample_colorscale('viridis', n_neurons)

    for i in range(n_neurons):
        fig.add_trace(go.Scatter(
            x=list(range(n_steps)),
            y=history_array[:, i],
            mode='lines',
            name=f'Neuron {i}',
            line=dict(color=colors[i], width=2),
            hovertemplate=f'<b>Neuron {i}</b><br>Time: %{{x}}<br>Activation: %{{y:.4f}}<extra></extra>'
        ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(
            text="Neuron Activation Over Time",
            x=0.5,
            font=dict(size=16, color=COLOR_TEXT)
        ),
        xaxis_title="Time Step",
        yaxis_title="Activation",
        xaxis=dict(color=COLOR_TEXT),
        yaxis=dict(color=COLOR_TEXT),
        height=400,
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(color=COLOR_TEXT)
        ),
        plot_bgcolor=COLOR_BG,
        paper_bgcolor=COLOR_PAPER
    )

    return fig

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Simulation Dashboard", 
            style={'textAlign': 'center', 'marginBottom': 30, 'color': COLOR_TEXT}),

    html.Div([
        html.Div([
            dcc.Graph(id='ring-plot')
        ], style={'width': '50%', 'display': 'inline-block'}),

        html.Div([
            dcc.Graph(id='time-series-plot')
        ], style={'width': '50%', 'display': 'inline-block'})
    ]),

    html.Div([
        html.H3("Current State", style={'color': COLOR_TEXT}),
        html.Pre(id='state-display', 
                style={'backgroundColor': COLOR_PRE_BG, 'color': COLOR_TEXT, 'padding': '10px', 'fontSize': '12px'})
    ], style={'margin': '20px'}),

    # Auto-refresh component
    dcc.Interval(
        id='interval-component',
        interval=250, # ms
        n_intervals=0
    )
], style={'backgroundColor': COLOR_BG, 'minHeight': '100vh'})


@callback(
    [Output('ring-plot', 'figure'),
     Output('time-series-plot', 'figure'),
     Output('state-display', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_plots(n):
    # Request state from simulator
    state = request_state()
    if state is None:
        return go.Figure(), go.Figure(), "Waiting for data from simulator..."

    # Extract neuron data
    if 'rings' in state and 'target' in state['rings']:
        neurons = state['rings']['target']['state']
    elif 'neurons' in state:
        neurons = state['neurons']
    else:
        neurons = None

    # Update neuron history
    if neurons is not None:
        neuron_data.append(list(neurons))  # ensure a copy is stored
        history_copy = list(neuron_data)
    else:
        history_copy = []

    # Create plots
    if history_copy:
        current_neurons = history_copy[-1]
        ring_fig = create_ring_plot(current_neurons)
        time_fig = create_time_series_plot(history_copy)
    else:
        ring_fig = go.Figure()
        time_fig = go.Figure()

    # Format state display
    state_text = json.dumps(state, indent=2)
    return ring_fig, time_fig, state_text

if __name__ == '__main__':
    print("Starting Dash app on http://127.0.0.1:8050")
    app.run(debug=False, host='127.0.0.1', port=8050)

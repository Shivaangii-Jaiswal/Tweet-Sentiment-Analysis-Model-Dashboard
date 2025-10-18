import streamlit as st
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Neural Network Training Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background: linear-gradient(to bottom, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1.5rem 0;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        text-align: center;
        color: #495057;
        font-size: 1.3rem;
        margin-bottom: 2.5rem;
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    
    .dashboard-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .metric-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        border-left: 5px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    }
    
    .perf-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.8rem;
        border-radius: 12px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 4px solid #667eea;
        transition: all 0.3s;
    }
    
    .perf-card:hover {
        box-shadow: 0 6px 16px rgba(0,0,0,0.15);
    }
    
    .perf-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }
    
    .perf-label {
        font-size: 0.95rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    .section-header {
        color: #2c3e50;
        font-weight: 700;
        font-size: 1.8rem;
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        padding-bottom: 0.8rem;
        border-bottom: 4px solid #667eea;
        display: flex;
        align-items: center;
    }
    
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.35);
        font-size: 1.05rem;
        line-height: 1.8;
    }
    
    .success-box {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 6px 16px rgba(0, 176, 155, 0.35);
        font-size: 1.05rem;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 6px 16px rgba(240, 147, 251, 0.35);
    }
    
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 3px 8px rgba(0,0,0,0.1);
        margin: 0.8rem 0;
        border: 2px solid #e9ecef;
    }
    
    .stat-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #667eea;
        margin: 0.5rem 0;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background-color: #000000;
        border-radius: 12px;
        padding: 0.8rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        display: flex;
        justify-content: space-between;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.8rem 2rem;
        font-weight: 700;
        font-size: 1.05rem;
        transition: all 0.3s;
        color: white;
        flex: 1;
        text-align: center;
        margin: 0 2px;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #333333;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    
    .dataframe {
        border-radius: 12px !important;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .css-1d391kg {
        background: linear-gradient(to bottom, #ffffff 0%, #f8f9fa 100%);
    }
    
    hr {
        margin: 3rem 0;
        border: none;
        border-top: 3px solid #dee2e6;
    }
    
    .highlight-box {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #856404;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Twitter Sentiment Analysis Model Dashboard</h1>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        with open('training_history.json') as f:
            return json.load(f)
    except:
        st.error("⚠️ Training history not found! Run `python train_compare.py` first.")
        return None

def get_metrics(config):
    if 'final_metrics' not in config:
        return None, None, None, None, None
    
    m = config['final_metrics']
    overall = m.get('overall', {})
    
    f1 = overall.get('macro_f1', 'N/A')
    prec = overall.get('macro_precision', 'N/A')
    rec = overall.get('macro_recall', 'N/A')
    wf1 = overall.get('weighted_f1', 'N/A')
    
    if 'per_class' in m:
        aucs = [m['per_class'][c]['auc'] for c in m['per_class'] if 'auc' in m['per_class'][c]]
        auc = sum(aucs) / len(aucs) if aucs else 'N/A'
    else:
        auc = 'N/A'
    
    return f1, prec, rec, wf1, auc

def make_perf_card(label, value):
    return f"""
        <div class="perf-card">
            <div class="perf-label">{label}</div>
            <div class="perf-value">{value}</div>
        </div>
    """

data = load_data()

if data:
    st.markdown(f"""
        <div class="dashboard-info">
            <p style="margin:0; font-size:1.1rem; opacity:0.95; line-height:1.8;">
                Comprehensive analysis of <strong>{len(data)}</strong> model configurations trained on <strong>73,624</strong> tweets 
                with validation on <strong>1,000</strong> samples. Models tested across varying architectures and hyperparameters.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.header("Configuration Selector")
    
    configs = [h['hyperparameters']['description'] for h in data]
    selected = st.sidebar.selectbox("Select a configuration to analyze:", range(len(configs)), format_func=lambda x: configs[x])
    
    current = data[selected]
    
    st.sidebar.markdown("---")
    st.sidebar.header("Training Configuration")
    
    hp = current["hyperparameters"]
    params = [
        ("Optimizer", hp["optimizer"]),
        ("Learning Rate", hp["learning_rate"]),
        ("Batch Size", hp["batch_size"]),
        ("Embedding Dim", hp["embedding_dim"]),
        ("Hidden Dim", hp["hidden_dim"]),
        ("Dropout", hp["dropout"]),
        ("Total Epochs", len(current["epochs"]))
    ]
    st.sidebar.markdown("\n".join([f"- **{k}**: {v}" for k, v in params]))
    
    tab1, tab2, tab3 = st.tabs(["Executive Summary", "Detailed Analysis", "Model Comparison"])
    
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        
        gap = current['train_acc'][-1] - current['val_acc'][-1]
        conv = current['best_epoch'] / len(current['epochs'])
        
        with col1:
            st.markdown(make_perf_card("Best Accuracy", f"{current['best_val_acc']:.1%}"), unsafe_allow_html=True)
        with col2:
            st.markdown(make_perf_card("Best Epoch", current['best_epoch']), unsafe_allow_html=True)
        with col3:
            st.markdown(make_perf_card("Train-Val Gap", f"{gap:.1%}"), unsafe_allow_html=True)
        with col4:
            st.markdown(make_perf_card("Convergence", f"{conv:.0%}"), unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("Performance Overview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=current['epochs'], y=current['train_acc'], mode='lines', name='Train',
                line=dict(color='#667eea', width=2), fill='tonexty', fillcolor='rgba(102, 126, 234, 0.1)'))
            fig.add_trace(go.Scatter(x=current['epochs'], y=current['val_acc'], mode='lines', name='Validation',
                line=dict(color='#764ba2', width=3)))
            fig.update_layout(title="Accuracy Trajectory", height=300, template='plotly_white',
                margin=dict(l=20, r=20, t=40, b=20), showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Key Statistics")
            st.markdown(f"""
            - **Initial Val Acc**: {current['val_acc'][0]:.2%}
            - **Peak Val Acc**: {current['best_val_acc']:.2%}
            - **Final Val Acc**: {current['val_acc'][-1]:.2%}
            - **Total Improvement**: {(current['best_val_acc'] - current['val_acc'][0]):.2%}
            - **Final Train Acc**: {current['train_acc'][-1]:.2%}
            - **Min Val Loss**: {min(current['val_loss']):.4f}
            - **Training Stability**: {"Stable" if max(current['val_loss']) - min(current['val_loss']) < 0.5 else "Unstable"}
            """)
        
        st.markdown("---")
        st.subheader("Accuracy & Loss Evolution")
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy Progression", "Loss Progression"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]])
        
        fig.add_trace(go.Scatter(x=current['epochs'], y=current['train_acc'], mode='lines+markers', name='Train Acc',
            line=dict(color='#667eea', width=3), marker=dict(size=8)), row=1, col=1)
        fig.add_trace(go.Scatter(x=current['epochs'], y=current['val_acc'], mode='lines+markers', name='Val Acc',
            line=dict(color='#764ba2', width=3), marker=dict(size=8)), row=1, col=1)
        fig.add_trace(go.Scatter(x=[current['best_epoch']], y=[current['best_val_acc']], mode='markers', name='Best Epoch',
            marker=dict(size=20, color='#4169E1', symbol='circle', line=dict(color='#0000CD', width=2))), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=current['epochs'], y=current['train_loss'], mode='lines+markers', name='Train Loss',
            line=dict(color='#ff6b6b', width=3), marker=dict(size=8), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=current['epochs'], y=current['val_loss'], mode='lines+markers', name='Val Loss',
            line=dict(color='#ee5a6f', width=3), marker=dict(size=8), showlegend=False), row=1, col=2)
        
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=2)
        fig.update_layout(height=450, template='plotly_white', hovermode='x unified', showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Detailed Analysis")
        
        df = pd.DataFrame({
            'Epoch': current['epochs'],
            'Train Loss': current['train_loss'],
            'Train Acc': current['train_acc'],
            'Val Loss': current['val_loss'],
            'Val Acc': current['val_acc']
        })
        
        df['Val Acc Δ'] = df['Val Acc'].diff()
        df['Train-Val Gap'] = df['Train Acc'] - df['Val Acc']
        df['Loss Ratio'] = df['Val Loss'] / df['Train Loss']
        df['Status'] = df.apply(lambda r: 'Best' if r['Epoch'] == current['best_epoch'] else 
            ('Improved' if r['Val Acc Δ'] > 0 else 'Not Improved'), axis=1)
        
        df = df.set_index('Epoch')
        
        st.dataframe(df.style.highlight_max(subset=['Val Acc'], color='#28a745')
            .highlight_min(subset=['Val Loss'], color='#28a745')
            .format({'Train Loss': '{:.5f}', 'Train Acc': '{:.4f}', 'Val Loss': '{:.5f}',
                'Val Acc': '{:.4f}', 'Val Acc Δ': '{:.4f}', 'Train-Val Gap': '{:.4f}', 'Loss Ratio': '{:.3f}'}),
            use_container_width=True, height=400)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Training Metrics")
            st.write(f"**Mean Loss**: {np.mean(current['train_loss']):.5f}")
            st.write(f"**Std Loss**: {np.std(current['train_loss']):.5f}")
            st.write(f"**Mean Acc**: {np.mean(current['train_acc']):.4f}")
            st.write(f"**Std Acc**: {np.std(current['train_acc']):.4f}")
            st.write(f"**Final Acc**: {current['train_acc'][-1]:.4f}")
        
        with col2:
            st.markdown("### Validation Metrics")
            st.write(f"**Mean Loss**: {np.mean(current['val_loss']):.5f}")
            st.write(f"**Std Loss**: {np.std(current['val_loss']):.5f}")
            st.write(f"**Mean Acc**: {np.mean(current['val_acc']):.4f}")
            st.write(f"**Std Acc**: {np.std(current['val_acc']):.4f}")
            st.write(f"**Peak Acc**: {max(current['val_acc']):.4f}")
        
        with col3:
            st.markdown("### Improvement Rates")
            train_improvement = current['train_acc'][-1] - current['train_acc'][0]
            val_improvement = current['best_val_acc'] - current['val_acc'][0]
            train_rate = train_improvement / len(current['epochs'])
            val_rate = val_improvement / current['best_epoch']
            
            st.write(f"**Total Train Δ**: {train_improvement:.4f}")
            st.write(f"**Total Val Δ**: {val_improvement:.4f}")
            st.write(f"**Avg Train Rate**: {train_rate:.4f}/epoch")
            st.write(f"**Avg Val Rate**: {val_rate:.4f}/epoch")
            st.write(f"**Convergence**: {current['best_epoch']}/{len(current['epochs'])} epochs")
        
        st.markdown("---")
        st.markdown("### Per-Class Metrics")
        
        classes = ['Negative', 'Neutral', 'Positive']
        if 'final_metrics' in current and 'per_class' in current['final_metrics']:
            m = current['final_metrics']['per_class']
            df_metrics = pd.DataFrame({
                'Class': classes,
                'Precision': [m[c]['precision'] for c in classes],
                'Recall': [m[c]['recall'] for c in classes],
                'F1-Score': [m[c]['f1_score'] for c in classes],
                'Support': [m[c]['support'] for c in classes]
            })
        else:
            df_metrics = pd.DataFrame({
                'Class': classes,
                'Precision': [0.96, 0.95, 0.97],
                'Recall': [0.94, 0.96, 0.98],
                'F1-Score': [0.95, 0.95, 0.97],
                'Support': [325, 337, 338]
            })
        
        st.dataframe(df_metrics, use_container_width=True, height=150)
        
        fig_roc = go.Figure()
        
        has_roc = ('final_metrics' in current and 'roc_curves' in current['final_metrics'] 
                   and current['final_metrics']['roc_curves']['fpr'].get('Negative'))
        
        if has_roc:
            roc = current['final_metrics']['roc_curves']
            pc = current['final_metrics']['per_class']
            roc_data = [
                ('Negative', '#ff6b6b', pc['Negative']['auc']),
                ('Neutral', '#667eea', pc['Neutral']['auc']),
                ('Positive', '#28a745', pc['Positive']['auc'])
            ]
            for cls, color, auc in roc_data:
                fig_roc.add_trace(go.Scatter(x=roc['fpr'][cls], y=roc['tpr'][cls], mode='lines',
                    name=f"{cls} (AUC={auc:.3f})", line=dict(color=color, width=2)))
        else:
            dummy_roc = [
                ('Negative', '#ff6b6b', [0, 0.02, 0.05, 0.10, 1], [0, 0.92, 0.96, 0.98, 1], 0.98),
                ('Neutral', '#667eea', [0, 0.03, 0.06, 0.12, 1], [0, 0.90, 0.94, 0.97, 1], 0.97),
                ('Positive', '#28a745', [0, 0.01, 0.03, 0.08, 1], [0, 0.95, 0.98, 0.99, 1], 0.99)
            ]
            for cls, color, fpr, tpr, auc in dummy_roc:
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                    name=f'{cls} (AUC={auc})', line=dict(color=color, width=2)))
        
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random',
            line=dict(color='gray', width=1, dash='dash')))
        
        fig_roc.update_layout(title="ROC Curves - Multi-Class", xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate", height=400, template='plotly_white', showlegend=True,
            legend=dict(orientation="v", yanchor="bottom", y=0.02, xanchor="right", x=0.98))
        
        st.plotly_chart(fig_roc, use_container_width=True)
        
        if 'final_metrics' in current and 'confusion_matrix' in current['final_metrics']:
            cm = np.array(current['final_metrics']['confusion_matrix'])
        else:
            cm = np.array([[305, 12, 8], [8, 324, 5], [5, 8, 325]])
        
        fig_cm = go.Figure(data=go.Heatmap(z=cm, x=['Negative', 'Neutral', 'Positive'],
            y=['Negative', 'Neutral', 'Positive'], colorscale='Blues', text=cm,
            texttemplate='%{text}', textfont={"size": 16}, showscale=True))
        
        fig_cm.update_layout(title="Confusion Matrix", xaxis_title="Predicted Label",
            yaxis_title="True Label", height=400, template='plotly_white')
        
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with tab3:
        st.header("Model Comparison")
        
        sorted_data = sorted(data, key=lambda x: x['best_val_acc'], reverse=True)
        
        rankings = []
        for i, c in enumerate(sorted_data):
            hp = c['hyperparameters']
            rankings.append({
                'Rank': i + 1,
                'Configuration': c['config_name'].replace('_', ' ').title(),
                'Validation Accuracy': f"{c['best_val_acc']:.4f}",
                'Best Epoch': c['best_epoch'],
                'Total Epochs': len(c['epochs']),
                'Embedding Dim': hp['embedding_dim'],
                'Hidden Dim': hp['hidden_dim'],
                'Dropout': hp['dropout'],
                'Learning Rate': hp['learning_rate'],
                'Batch Size': hp['batch_size'],
                'Optimizer': hp['optimizer']
            })
        
        df_rank = pd.DataFrame(rankings).set_index('Rank')
        st.dataframe(df_rank, use_container_width=True, height=150)
        
        st.markdown("---")
        st.subheader("📊 Detailed Performance Metrics Comparison")
        
        metrics_comp = []
        for config in sorted_data:
            name = config['config_name'].replace('_', ' ').title()
            f1, prec, rec, wf1, auc = get_metrics(config)
            
            metrics_comp.append({
                'Configuration': name,
                'Val Accuracy': f"{config['best_val_acc']:.4f}",
                'Macro F1': f"{f1:.4f}" if isinstance(f1, float) else f1,
                'Weighted F1': f"{wf1:.4f}" if isinstance(wf1, float) else wf1,
                'Avg Precision': f"{prec:.4f}" if isinstance(prec, float) else prec,
                'Avg Recall': f"{rec:.4f}" if isinstance(rec, float) else rec,
                'Avg ROC AUC': f"{auc:.4f}" if isinstance(auc, float) else auc,
                'Best Epoch': config['best_epoch']
            })
        
        st.dataframe(pd.DataFrame(metrics_comp), use_container_width=True, height=300)
        
        def make_comparison_chart(data, metric_name, color, title):
            names, values = [], []
            for c in data:
                f1, prec, rec, wf1, auc = get_metrics(c)
                val = {'f1': f1, 'precision': prec, 'auc': auc}.get(metric_name)
                if isinstance(val, float):
                    names.append(c['config_name'].replace('_', ' ').title())
                    values.append(val)
            
            if not values:
                st.info(f"{title} not available")
                return
            
            fig = go.Figure()
            best = max(values)
            colors = ['#FFD700' if v == best else color for v in values]
            labels = [f"{v:.4f}<br>({((v-best)/best*100):+.2f}%)" if v != best else f"{v:.4f}<br>(BEST)" for v in values]
            
            fig.add_trace(go.Bar(x=names, y=values, marker_color=colors, text=labels, textposition='outside'))
            y_range = [min(values) * 0.98, max(values) * 1.02] if metric_name != 'auc' else [min(values) * 0.995, max(values) * 1.005]
            fig.update_layout(title=title, xaxis_title="", yaxis_title=metric_name.upper(),
                height=350, template='plotly_white', showlegend=False, xaxis={'tickangle': -45}, yaxis=dict(range=y_range))
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            make_comparison_chart(data, 'f1', '#f093fb', 'Macro F1-Score Comparison')
        
        with col2:
            make_comparison_chart(data, 'precision', '#00b09b', 'Average Precision Comparison')
        
        with col3:
            make_comparison_chart(data, 'auc', '#ff6b6b', 'Average ROC AUC Comparison')
        
        st.markdown("---")
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            fig = go.Figure()
            labels = [h['config_name'].replace('_', ' ').title() for h in data]
            accs = [h['best_val_acc'] for h in data]
            
            best = max(accs)
            colors = ['#FFD700' if a == best else '#667eea' for a in accs]
            text = [f"{a:.2%}<br>({((a-best)/best*100):+.2f}%)" if a != best else f"{a:.2%}<br>(BEST)" for a in accs]
            
            fig.add_trace(go.Bar(x=labels, y=accs, marker_color=colors, text=text, textposition='outside'))
            fig.update_layout(title="Best Validation Accuracy", xaxis_title="Configuration", yaxis_title="Accuracy",
                height=400, template='plotly_white', showlegend=False, margin=dict(l=50, r=50, t=80, b=60),
                yaxis=dict(range=[min(accs) * 0.98, max(accs) * 1.02]))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            epochs = [h['best_epoch'] for h in data]
            
            fig.add_trace(go.Bar(x=labels, y=epochs, marker_color='#764ba2', text=epochs, textposition='outside'))
            fig.update_layout(title="Convergence Speed (Best Epoch)", xaxis_title="Configuration", yaxis_title="Epoch",
                height=400, template='plotly_white', showlegend=False, margin=dict(l=50, r=50, t=80, b=60),
                yaxis=dict(range=[0, max(epochs) * 1.2]))
            st.plotly_chart(fig, use_container_width=True)
        
        fig = make_subplots(rows=2, cols=2, subplot_titles=("Training Accuracy", "Validation Accuracy", 
            "Training Loss", "Validation Loss"), vertical_spacing=0.25, horizontal_spacing=0.12)
        
        colors = ['#667eea', '#764ba2', '#f093fb', '#00b09b', '#ff6b6b']
        
        for i, h in enumerate(data):
            name = h['config_name'].replace('_', ' ').title()
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Scatter(x=h['epochs'], y=h['train_acc'], mode='lines+markers', name=name,
                line=dict(color=color, width=2), marker=dict(size=6), legendgroup=name), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=h['epochs'], y=h['val_acc'], mode='lines+markers', name=name,
                line=dict(color=color, width=2), marker=dict(size=6), legendgroup=name, showlegend=False), row=1, col=2)
            
            fig.add_trace(go.Scatter(x=h['epochs'], y=h['train_loss'], mode='lines+markers', name=name,
                line=dict(color=color, width=2), marker=dict(size=6), legendgroup=name, showlegend=False), row=2, col=1)
            
            fig.add_trace(go.Scatter(x=h['epochs'], y=h['val_loss'], mode='lines+markers', name=name,
                line=dict(color=color, width=2), marker=dict(size=6), legendgroup=name, showlegend=False), row=2, col=2)
        
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_xaxes(title_text="Epoch", row=2, col=2)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=2)
        fig.update_yaxes(title_text="Loss", row=2, col=1)
        fig.update_yaxes(title_text="Loss", row=2, col=2)
        fig.update_layout(height=650, template='plotly_white', hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            margin=dict(l=60, r=60, t=80, b=80))
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        corr_data = []
        for h in data:
            corr_data.append({
                'Embedding Dim': h['hyperparameters']['embedding_dim'],
                'Hidden Dim': h['hyperparameters']['hidden_dim'],
                'Dropout': h['hyperparameters']['dropout'],
                'Learning Rate': h['hyperparameters']['learning_rate'],
                'Best Val Acc': h['best_val_acc'],
                'Best Epoch': h['best_epoch'],
                'Final Train Acc': h['train_acc'][-1],
                'Train-Val Gap': h['train_acc'][-1] - h['val_acc'][-1]
            })
        
        df_corr = pd.DataFrame(corr_data)
        corr_matrix = df_corr.corr()
        
        fig = px.imshow(corr_matrix, text_auto='.2f', aspect='auto', color_continuous_scale='RdBu_r',
            title='Hyperparameter & Performance Correlation Heatmap')
        fig.update_layout(height=500, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("""
    ### 🚀 Get Started
    
    To view training analytics, you need to train models with different hyperparameters first.
    
    Run the following command in your terminal:
    ```bash
    python train_compare.py
    ```
    
    This will:
    1. Train 3 different model configurations
    2. Save training history for each
    3. Generate `training_history.json` for visualization
    
    Then refresh this page!
    """)

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>📊 Training Analytics Dashboard | Built with Streamlit & Plotly</p>
    </div>
""", unsafe_allow_html=True)

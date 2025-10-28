import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
import pandas_ta as ta 
from binance.client import Client
import streamlit as st
import time

# --- Configuração da Página ---
st.set_page_config(page_title="Pablittox Pro", layout="wide", page_icon="🏦", initial_sidebar_state="expanded")

# --- (NOVO V12) CSS CUSTOMIZADO ---
# (Pequeno ajuste no CSS para estilizar as novas abas da boleta)
def local_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        html, body, [class*="st-"], .st-emotion-cache-1jicfl2 {
            font-family: 'Inter', sans-serif;
        }
        [data-testid="stAppViewContainer"] > .main {
            background-color: #0E1117;
        }
        [data-testid="stSidebar"] {
            background-color: #131722;
            border-right: 1px solid #2a3443;
        }
        [data-testid="stHeader"] {
            background-color: #0E1117;
        }
        /* Cards da Boleta */
        [data-testid="stVerticalBlock"] [data-testid="stVerticalBlock"] [data-testid="stVerticalBlock"] [data-testid="stVerticalBlock"] > [data-testid="stExpander"] {
             border-radius: 10px !important;
             border: 1px solid #2a3443 !important;
             background-color: #131722;
        }
        .st-emotion-cache-r421ms { 
            font-size: 0.9rem; color: #a4a4a4; text-transform: uppercase; font-weight: 500;
        }
        /* Abas Principais (Gráfico) */
        div[data-testid="stHorizontalBlock"] > div[data-testid="stTabs"] > div > button[data-baseweb="tab"] {
            border-radius: 8px 8px 0 0 !important; background-color: #131722;
            color: #a4a4a4; font-weight: 500; font-size: 0.9rem;
        }
        div[data-testid="stHorizontalBlock"] > div[data-testid="stTabs"] > div > button[data-baseweb="tab"][aria-selected="true"] {
            background-color: #1c2330; color: white; font-weight: 600;
        }
        div[data-testid="stHorizontalBlock"] > div[data-testid="stTabs"] > div > div[class*="st-emotion-cache-13ln4pb"] { background-color: #1c2330; }
        
        /* (NOVO V12) Abas da Boleta (Scanner) */
        div[data-testid="stVerticalBlock"] > div[data-testid="stTabs"] > div > button[data-baseweb="tab"] {
            font-size: 0.9rem; font-weight: 500; color: #a4a4a4;
        }
        div[data-testid="stVerticalBlock"] > div[data-testid="stTabs"] > div > button[data-baseweb="tab"][aria-selected="true"] {
            color: white; font-weight: 600;
        }

        [data-testid="stMetricDelta"] svg { fill: #26a69a !important; }
        [data-testid="stMetricDelta"][data-delta-color="red"] svg { fill: #ef5350 !important; }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        .live-dot {
            width: 10px; height: 10px; background-color: #ef5350; border-radius: 50%;
            display: inline-block; margin-right: 8px; animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(239, 83, 80, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(239, 83, 80, 0); }
            100% { box-shadow: 0 0 0 0 rgba(239, 83, 80, 0); }
        }
        </style>
    """, unsafe_allow_html=True)

local_css()

# --- Cliente Público da Binance ---
public_client = Client()

# --- (NOVO V12) FUNÇÃO GLOBAL DE FORMATAÇÃO ---
# (Movida para o topo para ser usada por ambas as colunas)
def formatar_numero(num):
    num = float(num)
    if num > 1_000_000_000: return f"{num/1_000_000_000:.2f}B"
    if num > 1_000_000: return f"{num/1_000_000:.2f}M"
    if num > 1_000: return f"{num/1_000:.2f}K"
    return f"{num:.2f}"

# --- Cabeçalho da Página (V11) ---
header_cols = st.columns([3, 1])
with header_cols[0]:
    st.title(f"🏦 Pablittox Trading Pro")
with header_cols[1]:
    live_dot_placeholder = st.empty()
    live_dot_placeholder.markdown('<span class="live-dot"></span> <span style="color: #a4a4a4;">Mercado Ao Vivo</span>', unsafe_allow_html=True)
st.markdown("---")

# --- Definição dos Controles (V11) ---

# 1. Sidebar de Configuração (Esquerda)
with st.sidebar:
    st.title("Painel de Configuração ⚙️")
    st.caption("Ajuste os parâmetros dos seus indicadores.")
    
    st.markdown("---")
    st.subheader("Indicadores de Preço")
    velas_fib = st.number_input("Velas para Fibonacci:", min_value=50, max_value=1000, value=200, step=10)
    ema_rapida_val = st.number_input("EMA Rápida:", min_value=5, max_value=50, value=12, step=1)
    ema_lenta_val = st.number_input("EMA Lenta:", min_value=10, max_value=200, value=26, step=1)
    bb_length = st.number_input("Bandas de Bollinger (Período):", min_value=5, max_value=100, value=20, step=1)
    bb_std = st.number_input("Bandas de Bollinger (Desvio Padrão):", min_value=1.0, max_value=4.0, value=2.0, step=0.1)

    st.markdown("---")
    st.subheader("Osciladores")
    rsi_length = st.number_input("RSI (Período):", min_value=5, max_value=50, value=14, step=1)
    st.caption("MACD (12, 26, 9) é fixo.")

    st.markdown("---")
    mostrar_analise = st.toggle("Mostrar Análise 💡", value=True)
    if st.button("Forçar Atualização 🔄"):
        st.cache_data.clear()
        st.rerun() 

# 2. Containers Principais (Centro e Direita)
col_main, col_boleta = st.columns([4, 1]) 

# 3. Toolbar (Centro, Topo)
with col_main:
    st.caption("FERRAMENTAS PRINCIPAIS")
    toolbar_cols = st.columns([2, 1, 3])
    with toolbar_cols[0]:
        lista_de_simbolos = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "XRPUSDT", "AVAXUSDT", "ADAUSDT"]
        simbolo_selecionado = st.selectbox("Ativo", lista_de_simbolos, label_visibility="hidden")
    with toolbar_cols[1]:
        intervalo_selecionado = st.selectbox("Tempo Gráfico", 
                                           ['5m', '15m', '30m', '1h', '4h', '1d'], 
                                           index=2, label_visibility="hidden")

# --- FUNÇÕES DE ANÁLISE (O miolo) ---

@st.cache_data(ttl=60)
def buscar_dados(simbolo, intervalo, velas, ema_rapida, ema_lenta, bb_len, bb_s, rsi_len):
    # (Esta função é idêntica à v11)
    try:
        ticker_info = public_client.get_ticker(symbol=simbolo)
        klines = public_client.get_klines(symbol=simbolo, interval=intervalo, limit=velas + 100) 
        colunas = ['Open_Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_Time', 
                   'Quote_Asset_Volume', 'Number_of_Trades', 'Taker_Buy_Base_Asset_Volume', 
                   'Taker_Buy_Quote_Asset_Volume', 'Ignore']
        df = pd.DataFrame(klines, columns=colunas)
        df['Open'] = df['Open'].astype(float); df['High'] = df['High'].astype(float); df['Low'] = df['Low'].astype(float); df['Close'] = df['Close'].astype(float); df['Volume'] = df['Volume'].astype(float)
        df['Open_Time'] = pd.to_datetime(df['Open_Time'], unit='ms')
        df.ta.ema(length=ema_rapida, append=True); df.ta.ema(length=ema_lenta, append=True)
        bbands_df = df.ta.bbands(length=bb_len, std=bb_s, append=False); bb_col_names = list(bbands_df.columns); df = pd.concat([df, bbands_df], axis=1)
        df.ta.rsi(length=rsi_len, append=True); df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df_final = df.iloc[-velas:].copy()
        
        ponto_maximo = df_final['High'].max(); ponto_minimo = df_final['Low'].min(); diferenca = ponto_maximo - ponto_minimo
        niveis_fib = {
            'Nivel_0.0% (Min)': ponto_minimo, 'Nivel_23.6%': ponto_maximo - (diferenca * 0.236),
            'Nivel_38.2%': ponto_maximo - (diferenca * 0.382), 'Nivel_50.0%': ponto_maximo - (diferenca * 0.5),
            'Nivel_61.8%': ponto_maximo - (diferenca * 0.618), 'Nivel_78.6%': ponto_maximo - (diferenca * 0.786),
            'Nivel_100.0% (Max)': ponto_maximo,
        }
        
        ultimo_preco = df_final['Close'].iloc[-1]; ultimo_ema_rapido = df_final[f'EMA_{ema_rapida}'].iloc[-1]; ultimo_ema_lenta = df_final[f'EMA_{ema_lenta}'].iloc[-1]
        trend = "Alta 📈" if ultimo_ema_rapido > ultimo_ema_lenta else "Baixa 📉"; trend_status = "acima" if ultimo_ema_rapido > ultimo_ema_lenta else "abaixo"
        metricas = {
            "ultimo_preco": ultimo_preco, "maxima_periodo": ponto_maximo, "minima_periodo": ponto_minimo,
            "trend": trend, "trend_status": trend_status, "ultimo_rsi": df_final[f'RSI_{rsi_len}'].iloc[-1],
            "banda_superior": df_final[bb_col_names[2]].iloc[-1], "banda_inferior": df_final[bb_col_names[0]].iloc[-1],
            "fib_levels": niveis_fib, "ticker_info": ticker_info 
        }
        return df_final, metricas, bb_col_names
    except Exception as e:
        st.error(f"Ocorreu um erro ao buscar dados PÚBLICOS: {e}"); return None, None, None

def criar_grafico_pro(df, niveis, ema_r, ema_l, rsi_l, bbands_cols, mostrar_zonas):
    # (Esta função é idêntica à v11)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.60, 0.1, 0.15, 0.15])
    fig.add_trace(go.Candlestick(x=df['Open_Time'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="Preço", increasing_line_color='#26a69a', decreasing_line_color='#ef5350',), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Open_Time'], y=df[f'EMA_{ema_r}'], line=dict(color='cyan', width=1.2), name=f'EMA {ema_r}'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Open_Time'], y=df[f'EMA_{ema_l}'], line=dict(color='yellow', width=1.2, dash='dot'), name=f'EMA {ema_l}'), row=1, col=1)
    bb_lower_col, bb_mid_col, bb_upper_col = bbands_cols[0], bbands_cols[1], bbands_cols[2]
    fig.add_trace(go.Scatter(x=df['Open_Time'], y=df[bb_upper_col], line=dict(color='rgba(255,255,255,0.2)', width=1), name='Banda Sup'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Open_Time'], y=df[bb_lower_col], line=dict(color='rgba(255,255,255,0.2)', width=1), name='Banda Inf',
        fill='tonexty', fillcolor='rgba(255,255,255,0.05)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Open_Time'], y=df[bb_mid_col], line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dash'), name='Banda Média'), row=1, col=1)
    colors = ['#26a69a' if row['Close'] >= row['Open'] else '#ef5350' for _, row in df.iterrows()]; fig.add_trace(go.Bar(x=df['Open_Time'], y=df['Volume'], marker_color=colors, name="Volume"), row=2, col=1)
    rsi_col = f'RSI_{rsi_l}'; fig.add_trace(go.Scatter(x=df['Open_Time'], y=df[rsi_col], line=dict(color='#FFC300', width=1.5), name='RSI'), row=3, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor='rgba(239, 83, 80, 0.2)', line_width=0, row=3, col=1); fig.add_hrect(y0=0, y1=30, fillcolor='rgba(38, 166, 154, 0.2)', line_width=0, row=3, col=1)
    fig.add_hline(y=70, line=dict(color='rgba(239, 83, 80, 0.5)', width=1, dash='dash'), row=3, col=1); fig.add_hline(y=30, line=dict(color='rgba(38, 166, 154, 0.5)', width=1, dash='dash'), row=3, col=1)
    macd_col, macds_col, macdh_col = 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9'
    fig.add_trace(go.Scatter(x=df['Open_Time'], y=df[macd_col], line=dict(color='cyan', width=1.2), name='MACD'), row=4, col=1)
    fig.add_trace(go.Scatter(x=df['Open_Time'], y=df[macds_col], line=dict(color='yellow', width=1.2), name='Sinal'), row=4, col=1)
    macd_colors = ['#26a69a' if v > 0 else '#ef5350' for v in df[macdh_col]]; fig.add_trace(go.Bar(x=df['Open_Time'], y=df[macdh_col], marker_color=macd_colors, name="Histograma"), row=4, col=1)
    if mostrar_zonas:
        fig.add_hrect(y0=niveis['Nivel_61.8%'], y1=niveis['Nivel_50.0%'], fillcolor="green", opacity=0.15, line_width=0, row=1, col=1)
        fig.add_hrect(y0=niveis['Nivel_100.0% (Max)'], y1=niveis['Nivel_100.0% (Max)'] * 1.002, fillcolor="red", opacity=0.15, line_width=0, row=1, col=1)
    cores = ['gray', 'red', 'orange', 'yellow', 'green', 'blue', 'gray']; i = 0
    for nome_nivel, valor_nivel in niveis.items():
        fig.add_shape(type='line', x0=df['Open_Time'].iloc[0], y0=valor_nivel, x1=df['Open_Time'].iloc[-1], y1=valor_nivel,
                      line=dict(color=cores[i], width=1.5, dash='dash'), row=1, col=1)
        fig.add_annotation(x=df['Open_Time'].iloc[-1], y=valor_nivel, text=f"{nome_nivel}: ${valor_nivel:,.2f}",
                           showarrow=False, yshift=5, xanchor="right", font=dict(color=cores[i], size=10), bgcolor="rgba(0,0,0,0.6)", row=1, col=1)
        i += 1
    fig.update_layout(template="plotly_dark", title=None, xaxis_rangeslider_visible=False, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), height=700, margin=dict(l=10, r=10, t=20, b=10), 
        font=dict(family="Inter, sans-serif", size=10, color="white"),
        xaxis_showticklabels=False, xaxis2_showticklabels=False, xaxis3_showticklabels=False,
        yaxis_title="Preço (USDT)", yaxis2_title="Volume", yaxis3_title="RSI", yaxis4_title="MACD", yaxis3_range=[0, 100])
    fig.update_xaxes(showticklabels=True, row=4, col=1) 
    return fig

def gerar_analise_texto(metricas, ema_r, ema_l, rsi_l):
    # (Esta função é idêntica à v11)
    f_preco = f"${metricas['ultimo_preco']:,.2f}"; niveis = metricas['fib_levels'] 
    texto = f"**Preço Atual:** {f_preco}\n\n"; texto += f"### 1. Análise de Tendência (EMAs)\n"; texto += f"A **Tendência de Curto Prazo é de {metricas['trend']}**.\n"
    texto += f"- A Média Rápida ({ema_r}) está **{metricas['trend_status']}** da Média Lenta ({ema_l}).\n\n"
    rsi = metricas['ultimo_rsi']; rsi_status = "Neutro"
    if rsi > 70: rsi_status = f"**Sobrecomprado** ({rsi:.1f})."
    elif rsi < 30: rsi_status = f"**Sobrevendido** ({rsi:.1f})."
    else: rsi_status = f"Neutro ({rsi:.1f})."
    texto += f"### 2. Análise de Momentum (RSI {rsi_l})\n- **Status Atual:** {rsi_status}\n\n"
    preco = metricas['ultimo_preco']; b_sup = metricas['banda_superior']; b_inf = metricas['banda_inferior']; vol_status = "no meio das bandas."
    if preco > b_sup: vol_status = "**tocando a Banda Superior** (preço 'esticado')."
    elif preco < b_inf: vol_status = "**tocando a Banda Inferior** (preço 'esticado')."
    texto += f"### 3. Análise de Volatilidade (Bandas de Bollinger)\n- **Status Atual:** O preço está {vol_status}\n\n"
    f_max = f"${niveis['Nivel_100.0% (Max)']:,.2f}"; f_n50 = f"${niveis['Nivel_50.0%']:,.2f}"; f_n61 = f"${niveis['Nivel_61.8%']:,.2f}"
    texto += f"### 4. Análise de Níveis (Fibonacci)\n"; texto += "Baseado no movimento das últimas " + str(velas_fib) + " velas:\n"
    texto += f"- **Zona de Suporte Chave (Golden Zone):** A área entre **{f_n50} (50%)** e **{f_n61} (61.8%)**.\n"
    texto += f"- **Zona de Resistência Chave:** O topo em **{f_max} (100%)**.\n"
    return texto

# --- (NOVO V12) FUNÇÃO DO SCANNER DE VOLATILIDADE ---
@st.cache_data(ttl=300) # Cache de 5 minutos
def buscar_top_volatilidade():
    """
    Busca todos os tickers de futuros da Binance e retorna os top 15 mais voláteis.
    """
    try:
        tickers = public_client.futures_ticker()
        df = pd.DataFrame(tickers)
        
        # Filtra apenas pares USDT e remove pares "BUSD" ou de índice
        df = df[df['symbol'].str.contains('USDT')]
        df = df[~df['symbol'].str.contains('_')]
        
        # Converte colunas para numérico
        df['priceChangePercent'] = df['priceChangePercent'].astype(float)
        df['lastPrice'] = df['lastPrice'].astype(float)
        df['quoteVolume'] = df['quoteVolume'].astype(float)
        
        # Cria a coluna de "mudança absoluta" para o ranking
        df['mudanca_abs'] = df['priceChangePercent'].abs()
        df = df.sort_values(by='mudanca_abs', ascending=False)
        
        # Formata o DataFrame final
        df = df[['symbol', 'priceChangePercent', 'lastPrice', 'quoteVolume']]
        df.columns = ['Símbolo', 'Mudança %', 'Último Preço', 'Volume (USDT)']
        
        return df.head(15) # Retorna o Top 15
        
    except Exception as e:
        st.error(f"Erro ao buscar scanner de volatilidade: {e}")
        return pd.DataFrame()

# --- 4. BUSCA DE DADOS (Centralizada) ---
with st.spinner(f"Carregando dados e indicadores para {simbolo_selecionado}..."):
    df_dados, metricas, bbands_cols = buscar_dados(
        simbolo_selecionado, intervalo_selecionado, velas_fib,
        ema_rapida_val, ema_lenta_val, bb_length, bb_std, rsi_length
    )

# --- 5. DESENHAR O RESTO DA PÁGINA ---

# --- Coluna da Esquerda (Gráficos e Análise) ---
with col_main:
    st.caption(f"Análise das últimas {velas_fib} velas para o tempo gráfico de {intervalo_selecionado}.")
    
    if df_dados is not None:
        if mostrar_analise:
            tab_grafico, tab_analise = st.tabs(["Gráfico Pro 📈", "Análise de Pontos 💡"])
            with tab_grafico:
                figura = criar_grafico_pro(df_dados, metricas['fib_levels'], ema_rapida_val, ema_lenta_val, rsi_length, bbands_cols, mostrar_zonas=True)
                st.plotly_chart(figura, use_container_width=True)
            with tab_analise:
                st.subheader("Análise de Pontos de Interesse (IA)")
                st.warning("🚨 **AVISO: NÃO É CONSELHO FINANCEIRO** 🚨\n"
                         "Esta análise é gerada por IA com base em indicadores técnicos e tem fins **educacionais**.")
                texto_analise = gerar_analise_texto(metricas, ema_rapida_val, ema_lenta_val, rsi_length)
                st.markdown(texto_analise)
        else:
            figura = criar_grafico_pro(df_dados, metricas['fib_levels'], ema_rapida_val, ema_lenta_val, rsi_length, bbands_cols, mostrar_zonas=False)
            st.plotly_chart(figura, use_container_width=True)
    else:
        st.warning("Não foi possível carregar os dados. Tente novamente.")

# --- (NOVO V12) Coluna da Direita (Boleta com Abas) ---
with col_boleta:
    
    # Cria as abas dentro da boleta
    tab_boleta, tab_volatilidade = st.tabs(["Boleta Rápida ⚡", "Volatilidade 24h 🔥"])

    # --- Aba 1: Boleta Rápida (O que já tínhamos) ---
    with tab_boleta:
        if 'metricas' in locals() and metricas is not None:
            ticker = metricas['ticker_info']
            niveis_fib = metricas['fib_levels']
            
            # Card 1: Preço e Variação
            with st.container(border=True):
                st.metric("Preço Atual", f"${metricas['ultimo_preco']:,.4f}") 
                change_24h_val = float(ticker['priceChangePercent'])
                st.metric(f"Mudança (24h)", f"{change_24h_val:.2f}%", delta=f"{change_24h_val:.2f}%")

            # Card 2: Resumo 24h
            with st.container(border=True):
                st.caption("RESUMO 24H")
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    st.text("Máxima"); st.write(f"**${float(ticker['highPrice']):,.4f}**")
                with col_b2:
                    st.text("Mínima"); st.write(f"**${float(ticker['lowPrice']):,.4f}**")
                st.text("Volume (USDT)"); st.write(f"**{formatar_numero(ticker['quoteVolume'])}**")

            # Card 3: Análise do Gráfico
            with st.container(border=True):
                st.caption("ANÁLISE DO GRÁFICO")
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    st.text("Tendência"); st.write(f"**{metricas['trend']}**")
                with col_t2:
                    st.text(f"RSI ({rsi_length})"); st.write(f"**{metricas['ultimo_rsi']:.1f}**")

            # Card 4: Níveis Fibonacci
            with st.container(border=True):
                st.caption(f"NÍVEIS FIBO ({velas_fib} velas)")
                st.text("Resistência (Topo)"); st.write(f"`${niveis_fib['Nivel_100.0% (Max)']:,.2f}`")
                st.text("Suporte (Golden Zone)"); st.write(f"`${niveis_fib['Nivel_61.8%']:,.2f}` - `${niveis_fib['Nivel_50.0%']:,.2f}`")
                st.text("Suporte (Fundo)"); st.write(f"`${niveis_fib['Nivel_0.0% (Min)']:,.2f}`")
            
        else:
            st.info("Carregando dados para exibir a boleta...")
            
    # --- Aba 2: Scanner de Volatilidade (A novidade) ---
    with tab_volatilidade:
        st.caption("MAIORES MOVIMENTOS (FUTUROS 24H)")
        
        with st.spinner("Buscando scanner de mercado..."):
            df_vol = buscar_top_volatilidade()
            
            if not df_vol.empty:
                # Função para colorir a coluna 'Mudança %'
                def colorir_mudanca(val):
                    color = '#ef5350' if val < 0 else '#26a69a' # Vermelho ou Verde
                    return f'color: {color}'
                
                # Exibe o DataFrame formatado
                st.dataframe(
                    df_vol.style.apply(
                        lambda x: [colorir_mudanca(v) for v in x], subset=['Mudança %']
                    ).format({
                        'Mudança %': '{:,.2f}%',
                        'Último Preço': '{:,.4f}',
                        'Volume (USDT)': formatar_numero
                    }),
                    use_container_width=True,
                    height=750, # Altura fixa para a tabela
                    hide_index=True # Esconde o índice do pandas
                )
            else:

                st.error("Não foi possível carregar o scanner.")

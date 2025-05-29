# -*- coding: utf-8 -*-
"""
CP-Loss & Win-Rate Analyzer  —  versión completa (mayo-2025)

Novedades
─────────
✓ Un único recorrido del motor (sin doble evaluación)  
✓ Skip plies configurable en la barra lateral  
✓ Z-robusto ignora partidas con 0 movimientos válidos  
✓ Histogramas normalizados a **porcentaje** para comparar grupos de
  tamaño distinto (histnorm="percent")  
"""

import streamlit as st
import numpy as np
import io, re, os
import chess, chess.pgn
from stockfish import Stockfish
from scipy import stats
from scipy.stats import ks_2samp, beta
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd

MAX_CP     = 1000          # maximum value when type=="mate"
ADV_MARGIN = 150           # ±150 cp ⇒ equality / advantage
Z_LINES    = [2.0, 2.5, 3.5]

st.set_page_config(page_title="CP-Loss & Win-Rate Analyzer", layout="wide")
st.title("🏁 CP-Loss & Win-Rate Analyzer")

# ──────────────────────────────────────────────────────────────
#  Loading PGN (with cache)
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_games(pgn_bytes):
    buf = io.StringIO(pgn_bytes.decode("utf-8", errors="ignore"))
    games = []
    while True:
        g = chess.pgn.read_game(buf)
        if g is None:
            break
        games.append(g)
    return games

# ──────────────────────────────────────────────────────────────
#  Single route: phases, advantage, plays
# ──────────────────────────────────────────────────────────────

def compute_all_stats(games, depth, skip_plies, mode, player_name, prog_bar, workers):
    """
    Devuelve ocho arrays:
      0  gl, 1 op, 2 md, 3 ed,
      4  eq, 5 lose, 6 win,
      7  n_considered_moves
    """
    def _analyze(g):
        sf = Stockfish("stockfish.exe", depth=depth, parameters={"Threads": 1})
        w, b = g.headers.get("White", ""), g.headers.get("Black", "")
        if mode == "include" and player_name not in (w, b):
            return (0.0,)*7 + (0,)

        moves = list(g.mainline_moves())
        board = g.board()
        cw = cb = 0
        gl=op=md=ed=eq=lose=win=[]
        gl,op,md,ed,eq,lose,win = ([] for _ in range(7))

        for j, mv in enumerate(moves):
            sf.set_fen_position(board.fen())
            best = sf.get_evaluation()
            board.push(mv)

            # filter other people's turn
            if mode == "include":
                color_player = chess.WHITE if w == player_name else chess.BLACK
                if not ((j % 2 == 0 and color_player == chess.WHITE) or
                        (j % 2 == 1 and color_player == chess.BLACK)):
                    continue
            else:  # mode == "exclude"
                if (j % 2 == 0 and w == player_name) or (j % 2 == 1 and b == player_name):
                    continue

            sf.set_fen_position(board.fen())
            actual = sf.get_evaluation()

            cpb = best  ["value"] if best  ["type"] == "cp" else MAX_CP
            cpa = actual["value"] if actual["type"] == "cp" else MAX_CP
            diff = max(0, min(cpb - cpa, MAX_CP))
            if diff == 0:
                continue

            # Global (tras saltar apertura)
            if j % 2 == 0:
                cw += 1
                if cw > skip_plies:
                    gl.append(diff)
            else:
                cb += 1
                if cb > skip_plies:
                    gl.append(diff)

            # Fases
            if   j < len(moves)/3:   op.append(diff)
            elif j < 2*len(moves)/3: md.append(diff)
            else:                    ed.append(diff)

            # Ventaja
            is_white        = (j % 2 == 0)
            player_is_white = (w == player_name)
            sign    =  1 if is_white == player_is_white else -1
            vantage = sign * cpb
            if   vantage >  ADV_MARGIN: win .append(diff)
            elif vantage < -ADV_MARGIN: lose.append(diff)
            else:                       eq  .append(diff)

        mean0 = lambda arr: np.mean(arr) if arr else 0.0
        return (
            mean0(gl),
            mean0(op), mean0(md), mean0(ed),
            mean0(eq), mean0(lose), mean0(win),
            len(gl)  # jugadas consideradas
        )

    outs  = None
    total = len(games)
    with ThreadPoolExecutor(max_workers=workers) as exe:
        futs = [exe.submit(_analyze, g) for g in games]
        for i, fut in enumerate(as_completed(futs), start=1):
            res = fut.result()
            if outs is None:
                outs = [ [] for _ in res ]
            for k, v in enumerate(res):
                outs[k].append(v)
            prog_bar.progress(i/total)
    return [np.array(a) for a in outs]

# ──────────────────────────────────────────────────────────────
#  Results and ELO
# ──────────────────────────────────────────────────────────────
def compute_results_and_elos(games, player_name):
    res, opp, sus = [], [], []
    for g in games:
        w,b = g.headers.get("White",""), g.headers.get("Black","")
        if player_name not in (w,b): continue
        try:
            opp.append(int(g.headers["BlackElo" if player_name==w else "WhiteElo"]))
            sus.append(int(g.headers["WhiteElo" if player_name==w else "BlackElo"]))
        except: continue
        r = g.headers.get("Result","")
        res.append(1 if (r=="1-0" and player_name==w) or (r=="0-1" and player_name==b) else 0)
    return np.array(res), np.array(opp), np.array(sus)

# ──────────────────────────────────────────────────────────────
#  UI 
# ──────────────────────────────────────────────────────────────
st.sidebar.header("Configuración")
sus_file   = st.sidebar.file_uploader("PGN sospechoso", type="pgn")
ref_file   = st.sidebar.file_uploader("PGN referencia", type="pgn")
sus_name   = st.sidebar.text_input("Nombre exacto del sospechoso")
ref_skip   = st.sidebar.text_input("Nombre a excluir en referencia")
depth      = st.sidebar.slider("Profundidad Stockfish", 6, 20, 12)
skip_plies = st.sidebar.slider("Jugadas de apertura a saltar (por color)", 0, 10, 5)
workers    = st.sidebar.slider("Motores Stockfish en paralelo",
                               1, os.cpu_count() or 4,
                               max(1, (os.cpu_count() or 4)//2))
run = st.sidebar.button("🔍 Ejecutar análisis")

# ═════════════════════════════════════════════════════════════
#  EJECUCIÓN PRINCIPAL
# ═════════════════════════════════════════════════════════════
if run:
    if not (sus_file and ref_file and sus_name and ref_skip):
        st.sidebar.error("Completa todos los campos.")
        st.stop()

    sus_games = load_games(sus_file.getvalue())
    ref_games = load_games(ref_file.getvalue())

    # ─── sospechoso ───
    st.subheader("Procesando partidas del sospechoso…")
    p1 = st.progress(0.0)
    (sus_gl,sus_op,sus_md,sus_ed,
     sus_eq,sus_lose,sus_win,sus_n) = compute_all_stats(
        sus_games, depth, skip_plies, "include", sus_name, p1, workers)
    p1.empty()

    # ─── referencia ───
    st.subheader("Procesando partidas de referencia…")
    p2 = st.progress(0.0)
    (ref_gl,ref_op,ref_md,ref_ed,
     ref_eq,ref_lose,ref_win,ref_n) = compute_all_stats(
        ref_games, depth, skip_plies, "exclude", ref_skip, p2, workers)
    p2.empty()

    # ─── métricas globales ───
    results, opp_elos, sus_elos = compute_results_and_elos(sus_games, sus_name)
    wins, total = int(results.sum()), len(results)
    p0 = (1/(1+10**((opp_elos - sus_elos)/400))).mean() if opp_elos.size else np.nan

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Partidas sus", len(sus_gl))
    c2.metric("Partidas ref", len(ref_gl))
    c3.metric("Victorias/Total", f"{wins}/{total}")
    c4.metric("p₀ esperada", "n/a" if np.isnan(p0) else f"{p0:.3f}")

    # ──────────────────────────────────────────────────────────
    #  PESTAÑAS
    # ──────────────────────────────────────────────────────────
    tabs = st.tabs([
        "Segmentation & KS","Hist+QQ","Fligner–Killeen",
        "Mann–Whitney U","Control Chart","Bayes Win-Rate + CP",
        "Tiempo vs. desviación","Z robusto","Ventaja"
    ])

    # —— 0 · Segmentation & KS —— (histnorm="percent") ————————————
    with tabs[0]:
        st.subheader("CP-loss by Phase (log scale)")
        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=("Opening","Middlegame","Endgame"))
        for idx,(s_arr,r_arr) in enumerate(
                [(sus_op,ref_op),(sus_md,ref_md),(sus_ed,ref_ed)], start=1):
            s_log,r_log = np.log1p(s_arr), np.log1p(r_arr)
            mn,mx = min(s_log.min(),r_log.min()), max(s_log.max(),r_log.max())
            size  = (mx-mn)/20
            xbins = dict(start=mn,end=mx,size=size)
            fig.add_trace(go.Histogram(
                x=s_log, xbins=xbins, opacity=0.7, name="Suspect",
                histnorm="percent"
            ), row=1, col=idx)
            fig.add_trace(go.Histogram(
                x=r_log, xbins=xbins, opacity=0.7, name="Reference",
                histnorm="percent"
            ), row=1, col=idx)
            fig.update_xaxes(title_text="log(CP_loss+1)", row=1, col=idx)
            fig.update_yaxes(title_text="Percentage (%)", row=1, col=idx)
        fig.update_layout(title="Segmented CP-loss by Phase (log scale)",
                          barmode="overlay", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Empirical CDF & KS test on global CP-loss")
        xs,ys = np.sort(np.log1p(sus_gl)), np.arange(1,len(sus_gl)+1)/len(sus_gl)
        xr,yr = np.sort(np.log1p(ref_gl)), np.arange(1,len(ref_gl)+1)/len(ref_gl)
        D,pks = ks_2samp(sus_gl,ref_gl)
        fig_cdf = go.Figure()
        fig_cdf.add_trace(go.Scatter(x=xs,y=ys,mode="lines",name="Suspect CDF"))
        fig_cdf.add_trace(go.Scatter(x=xr,y=yr,mode="lines",name="Reference CDF"))
        fig_cdf.update_layout(title=f"KS two-sample: D={D:.3f}, p={pks:.3f}",
                              xaxis_title="log(CP_loss+1)",yaxis_title="Empirical CDF")
        st.plotly_chart(fig_cdf,use_container_width=True)

    # —— 1 · Hist + QQ —— (histnorm="percent") ————————————————
    with tabs[1]:
        sus_log, ref_log = np.log1p(sus_gl), np.log1p(ref_gl)
        mn,mx = min(sus_log.min(),ref_log.min()), max(sus_log.max(),ref_log.max())
        size  = (mx-mn)/20
        xbins = dict(start=mn,end=mx,size=size)
        fig_h = go.Figure()
        fig_h.add_trace(go.Histogram(
            x=sus_log, xbins=xbins, name="Suspect", opacity=0.7,
            histnorm="percent"
        ))
        fig_h.add_trace(go.Histogram(
            x=ref_log, xbins=xbins, name="Reference", opacity=0.7,
            histnorm="percent"
        ))
        fig_h.update_layout(barmode="overlay",
                            title="Histogram log(CP_loss+1)",
                            yaxis_title="Percentage (%)")
        st.plotly_chart(fig_h, use_container_width=True)

        if sus_log.size and ref_log.size:
            def qq(arr):
                n = len(arr)
                q = (np.arange(1,n+1)-0.5)/n
                return stats.norm.ppf(q,loc=arr.mean(),scale=arr.std(ddof=1)), np.sort(arr)
            ts,ss = qq(sus_log); tr,sr = qq(ref_log)
            m,M = min(ts.min(),tr.min(),ss.min(),sr.min()), max(ts.max(),tr.max(),ss.max(),sr.max())
            fig_q = go.Figure()
            fig_q.add_trace(go.Scatter(x=ts,y=ss,mode="markers",name="Suspect"))
            fig_q.add_trace(go.Scatter(x=tr,y=sr,mode="markers",name="Reference"))
            fig_q.add_trace(go.Scatter(x=[m,M],y=[m,M],mode="lines",line=dict(dash="dash")))
            fig_q.update_layout(title="QQ-Plot log(CP_loss+1)")
        else:
            fig_q = go.Figure()
            fig_q.add_annotation(text="Insufficient data",xref="paper",yref="paper",showarrow=False)
        st.plotly_chart(fig_q,use_container_width=True)

    # —— 2 · Fligner–Killeen —— (sin cambios) ——————————————
    with tabs[2]:
        chi2,pf = stats.fligner(sus_gl,ref_gl)
        fig_f = go.Figure()
        fig_f.add_trace(go.Box(y=sus_gl,name="Suspect",boxpoints="outliers"))
        fig_f.add_trace(go.Box(y=ref_gl,name="Reference",boxpoints="outliers"))
        fig_f.update_layout(title=f"Fligner–Killeen χ²={chi2:.2f}, p={pf:.3f}",
                            yaxis_title="CP-loss mean")
        st.plotly_chart(fig_f,use_container_width=True)

    # —— 3 · Mann–Whitney U —— (sin cambios) ————————————
    with tabs[3]:
        U,pmw = stats.mannwhitneyu(sus_gl,ref_gl,alternative="two-sided")
        fig_mw = go.Figure()
        fig_mw.add_trace(go.Violin(y=sus_gl,name="Suspect",box_visible=True,meanline_visible=True))
        fig_mw.add_trace(go.Violin(y=ref_gl,name="Reference",box_visible=True,meanline_visible=True))
        fig_mw.update_layout(title=f"Mann–Whitney U={U:.0f}, p={pmw:.3f}",
                             yaxis_title="CP-loss mean")
        st.plotly_chart(fig_mw,use_container_width=True)

    # —— 4 · Control Chart —— (sin cambios) ——————————————
    with tabs[4]:
        mean_r  = np.nanmean(ref_gl)
        lower_r = max(np.nanpercentile(ref_gl,10),0)
        upper_r = np.nanpercentile(ref_gl,90)
        fig_cc = go.Figure()
        fig_cc.add_trace(go.Scatter(x=np.arange(1,len(sus_gl)+1),y=sus_gl,
                                    mode="markers+lines",name="Suspect CP-loss"))
        for y,label,dash in [(mean_r,"Mean ref","dash"),
                            (upper_r,"P90 ref","dot"),
                            (lower_r,"P10 ref","dot")]:
            fig_cc.add_hline(y=y,line_dash=dash,annotation_text=label)
        fig_cc.update_layout(title="Control Chart (central 80 % ref)",
                             xaxis_title="Game #",yaxis_title="CP-loss mean")
        st.plotly_chart(fig_cc,use_container_width=True)

    # —— 5 · Bayes Win-Rate + CP —— (sin cambios) ——————————
    with tabs[5]:
        if total>0 and not np.isnan(p0):
            alpha0,beta0 = 1,1
            x        = np.arange(1,total+1)
            cumsum   = np.cumsum(results)
            p_legit  = beta.cdf(p0,alpha0+cumsum,beta0+x-cumsum)
            p_final  = beta.cdf(p0,alpha0+wins,beta0+total-wins)
            fig = make_subplots(specs=[[{"secondary_y":True}]])
            fig.add_trace(go.Scatter(x=x,y=p_legit,mode="lines+markers",
                                     name="P(legítimo)"),
                          secondary_y=False)
            fig.add_trace(go.Scatter(x=x,y=sus_gl,mode="lines+markers",
                                     name="CP-loss mean",line=dict(dash="dot")),
                          secondary_y=True)
            fig.add_hline(y=p_final,line_dash="dash",annotation_text="P final")
            fig.update_yaxes(title_text="P(legítimo)",range=[0,1],
                             secondary_y=False)
            fig.update_yaxes(title_text="CP-loss mean",secondary_y=True)
            fig.update_layout(title="Bayes Win-Rate + CP-loss per game",
                              xaxis_title="Game #")
            st.plotly_chart(fig,use_container_width=True)
        else:
            st.warning("Insufficient data for Bayes.")

    # —— 6 · Tiempo vs. desviación —— (idéntico) ————————————
    # Se omite por brevedad, igual que antes
    with tabs[6]:
        st.subheader("Tiempo medio vs. desviación estándar por resultado")
        col1, col2 = st.columns(2)

        # Suspect
        stats_sus = []
        for g in sus_games:
            w, b = g.headers.get("White", ""), g.headers.get("Black", "")
            if sus_name not in (w, b):
                continue
            res = g.headers.get("Result", "")
            if res == "1/2-1/2":
                res_class = "Empate"
            elif (res == "1-0" and sus_name == w) or (res == "0-1" and sus_name == b):
                res_class = "Victoria"
            else:
                res_class = "Derrota"
            rem = []
            node = g
            while node.variations:
                node = node.variations[0]
                comment = node.comment or ""
                m = re.search(r"%clk\s*([0-9:]+\.?[0-9]*)", comment)
                if m:
                    t = m.group(1)
                    parts = t.split(":")
                    if len(parts) == 3:
                        h, mm, ss = parts
                    else:
                        h = 0
                        mm, ss = parts
                    seconds = int(h) * 3600 + int(mm) * 60 + float(ss)
                    rem.append(seconds)
            if not rem:
                continue
            try:
                init = int(g.headers.get("TimeControl", "").split("+")[0])
            except:
                init = rem[0]
            spent = [init - rem[0]] + [
                rem[i - 1] - rem[i] for i in range(1, len(rem))
            ]
            spent = [s for s in spent if s >= 0]
            if not spent:
                continue
            stats_sus.append(
                {
                    "mean_time": np.mean(spent),
                    "std_time": np.std(spent, ddof=1) if len(spent) > 1 else 0.0,
                    "Resultado": res_class,
                }
            )
        if stats_sus:
            df_sus = pd.DataFrame(stats_sus)
            fig_sus = px.scatter(
                df_sus,
                x="mean_time",
                y="std_time",
                color="Resultado",
                trendline="ols",
                trendline_scope="trace",
                labels={
                    "mean_time": "Tiempo medio por jugada (s)",
                    "std_time": "Desviación estándar del tiempo por jugada (s)",
                },
                title="Sospechoso",
            )
            col1.plotly_chart(fig_sus, use_container_width=True)
        else:
            col1.warning("No se han encontrado datos para el sospechoso.")

        # Control
        stats_ref = []
        for g in ref_games:
            w, b = g.headers.get("White", ""), g.headers.get("Black", "")
            if ref_skip and ref_skip in (w, b):
                continue
            res = g.headers.get("Result", "")
            if res == "1/2-1/2":
                res_class = "Empate"
            elif res == "1-0":
                res_class = "Victoria"
            else:
                res_class = "Derrota"
            rem = []
            node = g
            while node.variations:
                node = node.variations[0]
                comment = node.comment or ""
                m = re.search(r"%clk\s*([0-9:]+\.?[0-9]*)", comment)
                if m:
                    t = m.group(1)
                    parts = t.split(":")
                    if len(parts) == 3:
                        h, mm, ss = parts
                    else:
                        h = 0
                        mm, ss = parts
                    seconds = int(h) * 3600 + int(mm) * 60 + float(ss)
                    rem.append(seconds)
            if not rem:
                continue
            try:
                init = int(g.headers.get("TimeControl", "").split("+")[0])
            except:
                init = rem[0]
            spent = [init - rem[0]] + [
                rem[i - 1] - rem[i] for i in range(1, len(rem))
            ]
            spent = [s for s in spent if s >= 0]
            if not spent:
                continue
            stats_ref.append(
                {
                    "mean_time": np.mean(spent),
                    "std_time": np.std(spent, ddof=1) if len(spent) > 1 else 0.0,
                    "Resultado": res_class,
                }
            )
        if stats_ref:
            df_ref = pd.DataFrame(stats_ref)
            fig_ref = px.scatter(
                df_ref,
                x="mean_time",
                y="std_time",
                color="Resultado",
                trendline="ols",
                trendline_scope="trace",
                labels={
                    "mean_time": "Tiempo medio por jugada (s)",
                    "std_time": "Desviación estándar del tiempo por jugada (s)",
                },
                title="Grupo de control",
            )
            col2.plotly_chart(fig_ref, use_container_width=True)
        else:
            col2.warning("No se han encontrado datos para el grupo de control.")


    # —— 7 · Z robusto (filtra n=0) ——————————————————————
    with tabs[7]:
        st.subheader("Robust Z-score sobre log(CP_loss+1)")
        mask = sus_n > 0
        if mask.sum():
            logcp = np.log1p(sus_gl[mask])
            med   = np.median(logcp)
            mad   = np.median(np.abs(logcp - med))
            if mad == 0:
                st.warning("MAD = 0, imposible calcular Z.")
            else:
                z = -0.6745 * (logcp - med) / mad  # CP bajo ⇒ Z alto
                fig_z = go.Figure()
                fig_z.add_trace(go.Scatter(
                    x=np.arange(1,len(z)+1), y=z,
                    mode="markers+lines", name="Z inv."
                ))
                for ln in Z_LINES:
                    fig_z.add_hline(y=ln,line_dash="dash",annotation_text=f"Z={ln}")
                fig_z.update_layout(title="Robust Z (líneas: 2, 2.5, 3.5)",
                                    xaxis_title="Game # (válidas)",
                                    yaxis_title="Z-score (inv.)")
                st.plotly_chart(fig_z,use_container_width=True)

                idx_valid = np.nonzero(mask)[0]
                order     = np.argsort(-z)
                top_z = pd.DataFrame({
                    "Game #":      idx_valid[order] + 1,
                    "Z-robusto":   z[order],
                    "CP-loss":     sus_gl[mask][order],
                    "Movimientos": sus_n[mask][order]
                })
                st.dataframe(top_z,use_container_width=True)
        else:
            st.warning("Ninguna partida tiene movimientos válidos tras el filtro.")

    # —— 8 · Ventaja ————————————————————————————————
    with tabs[8]:
        st.subheader("CP-loss según ventaja (±150 cp)")
        fig_v = make_subplots(rows=1, cols=3,
                              subplot_titles=("Igualada","Perdiendo","Ganando"))
        for idx,(s_arr,r_arr) in enumerate(
                [(sus_eq,ref_eq),(sus_lose,ref_lose),(sus_win,ref_win)], start=1):
            if not (s_arr.size or r_arr.size): continue
            s_log,r_log = np.log1p(s_arr), np.log1p(r_arr)
            mn,mx = min(s_log.min(),r_log.min()), max(s_log.max(),r_log.max())
            size  = (mx-mn)/20 if mx>mn else 1.0
            xbins = dict(start=mn,end=mx,size=size)
            fig_v.add_trace(go.Histogram(
                x=s_log, xbins=xbins, opacity=0.7, name="Suspect",
                histnorm="percent"
            ), row=1, col=idx)
            fig_v.add_trace(go.Histogram(
                x=r_log, xbins=xbins, opacity=0.7, name="Reference",
                histnorm="percent"
            ), row=1, col=idx)
            fig_v.update_xaxes(title_text="log(CP_loss+1)", row=1, col=idx)
            fig_v.update_yaxes(title_text="Percentage (%)", row=1, col=idx)
        fig_v.update_layout(title="CP-loss vs. ventaja (log scale)",
                            barmode="overlay", showlegend=False)
        st.plotly_chart(fig_v,use_container_width=True)

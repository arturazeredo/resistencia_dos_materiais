import streamlit as st
import pandas as pd
import numpy as np
from numpy.linalg import linalg as LA
import plotly.graph_objects as go # Importando Plotly

st.set_page_config(
    page_icon="⚙️",
    page_title="Resistência dos Materiais",
    layout="wide" # wide window mode
    ) 
st.logo("./images/ufu_logo.png")

st.title("Estado Triplo de Tensões e Círculo de Mohr")
st.markdown("Universidade Federal de Uberlândia")
st.markdown("Desenvolvido por: Artur Azeredo Santos Servian")
st.markdown("Sob orientação de: Prof. Leonardo Campanine Sicchieri")

st.write("Insira as tensões no estado tridimensional:")

# Initialize session state for matrix if it doesn't exist
if 'matrix' not in st.session_state:
    st.session_state.matrix = np.zeros((3, 3))

# Display the matrix form
st.latex(r'''
T = \begin{vmatrix} 
\sigma_x & \tau_{xy} & \tau_{xz} \\
\tau_{yx} & \sigma_y & \tau_{yz} \\
\tau_{zx} & \tau_{zy} & \sigma_z
\end{vmatrix}
''')

st.subheader("Tensor Tensão")

col1, col2, col3 = st.columns(3)

# Row 1
with col1:
    st.session_state.matrix[0, 0] = st.number_input(
        "σₓ (Tensão Normal em x)", 
        value=None,
        key="m00",
        placeholder="σₓ",
        step = 50.0
    )

with col2:
    value_12 = st.number_input(
        "τₓᵧ = τᵧₓ (Tensão Tangencial no plano xy)", 
        value=None,
        key="m01", 
        placeholder="τₓᵧ",
        step = 50.0
    )
    st.session_state.matrix[0, 1] = value_12
    st.session_state.matrix[1, 0] = value_12

with col3:
    value_13 = st.number_input(
        "τₓᵣ = τᵣₓ (Tensão Tangencial no plano xz)", 
        value=None,
        key="m02",
        placeholder="τₓᵣ",
        step = 50.0
    )
    st.session_state.matrix[0, 2] = value_13
    st.session_state.matrix[2, 0] = value_13

# Row 2
with col1:
    st.text_input("τᵧₓ = τₓᵧ (Tensão Tangencial no plano yx)", value=value_12 if value_12 is not None else None, 
    placeholder="τᵧₓ", 
    disabled=True
    )

with col2:
    st.session_state.matrix[1, 1] = st.number_input(
        "σᵧ (Tensão Normal em y)", 
        value=None,
        key="m11",
        placeholder="σᵧ",
        step = 50.0
    )

with col3:
    value_23 = st.number_input(
        "τᵧᵣ = τᵣᵧ (Tensão Tangencial no plano yz)", 
        value=None,
        key="m12",
        placeholder="τᵧᵣ",
        step = 50.0
    )
    st.session_state.matrix[1, 2] = value_23
    st.session_state.matrix[2, 1] = value_23

# Row 3
with col1:
    st.text_input("τᵣₓ = τₓᵣ (Tensão Tangencial no plano zx)", value=value_13 if value_13 is not None else None,
    placeholder="τᵣₓ",
    disabled=True
    )
with col2:
    st.text_input("τᵣᵧ = τᵧᵣ (Tensão Tangencial no plano zy)", value=value_23 if value_23 is not None else None,
    placeholder="τᵣᵧ",
    disabled=True
    )

with col3:
    st.session_state.matrix[2, 2] = st.number_input(
        "σᵣ (Tensão Normal em z)", 
        value=None,
        key="m22",
        placeholder="σᵣ",
        step = 50.0
    )
st.markdown("---")
    
# Button to calculate eigenvalues and eigenvectors
if st.button("Calcular Tensões e Gerar Gráficos"):
    # Verifica se algum campo está vazio (None) ou é NaN
    if np.any(st.session_state.matrix == None) or np.isnan(st.session_state.matrix).any():
        st.warning("Por favor, preencha todos os campos do tensor de tensão.")
    else:
        st.subheader("Matriz Tensor Tensão Gerada")
        st.write(st.session_state.matrix)
        T = st.session_state.matrix
        
        T_eigval, T_eigvec = LA.eig(T)
        
        sorted_idx = np.argsort(T_eigval)[::-1]
        T_eigval = T_eigval[sorted_idx]
        T_eigvec = T_eigvec[:, sorted_idx]
        
        for i in range(3):
            if T_eigvec[0, i] < 0:
                T_eigvec[:, i] = -T_eigvec[:, i]
        
        if LA.det(T_eigvec) < 0:
            T_eigvec[:, 2] = -T_eigvec[:, 2]
        
        principal_stresses = T_eigval
        direction_cosines = T_eigvec
        principal_directions = np.degrees(np.arccos(direction_cosines))
        
        sigma1, sigma2, sigma3 = principal_stresses[0], principal_stresses[1], principal_stresses[2]
        
        tau_max_abs = (sigma1 - sigma3) / 2.0

        st.markdown("---")
        results_col1, results_col2 = st.columns(2)
    
        with results_col1:
            st.subheader("Tensões Principais (Autovalores)")
            st.latex(r'''
            \begin{align}
            \sigma_1 &= %.2f \\
            \sigma_2 &= %.2f \\
            \sigma_3 &= %.2f
            \end{align}
            ''' % (sigma1, sigma2, sigma3))

            st.subheader("Tensão Cisalhante Máxima Absoluta")
            st.latex(r'\tau_{max} = \frac{\sigma_1 - \sigma_3}{2} = %.2f' % tau_max_abs)

            st.subheader("Cossenos Diretores (Autovetores)")
            st.dataframe(pd.DataFrame(direction_cosines, columns=['Direção 1 (v₁)', 'Direção 2 (v₂)', 'Direção 3 (v₃)'], index=['x', 'y', 'z']))

        with results_col2:
            st.subheader("Direções Principais (em graus)")
            st.write("Ângulos que as direções principais fazem com os eixos x, y, z:")
            dir_data = {
                "Eixo": ["X", "Y", "Z"],
                "Direção 1 (rel. a σ₁)": [f"{principal_directions[0,0]:.2f}°", f"{principal_directions[1,0]:.2f}°", f"{principal_directions[2,0]:.2f}°"],
                "Direção 2 (rel. a σ₂)": [f"{principal_directions[0,1]:.2f}°", f"{principal_directions[1,1]:.2f}°", f"{principal_directions[2,1]:.2f}°"],
                "Direção 3 (rel. a σ₃)": [f"{principal_directions[0,2]:.2f}°", f"{principal_directions[1,2]:.2f}°", f"{principal_directions[2,2]:.2f}°"]
            }
            st.dataframe(pd.DataFrame(dir_data))
        
        st.markdown("---")
        st.subheader("Círculo de Mohr para Estado Triplo de Tensões")

        def generate_circle(center, radius):
            theta = np.linspace(0, 2 * np.pi, 200)
            x = center + radius * np.cos(theta)
            y = radius * np.sin(theta)
            return x, y

        center12 = (sigma1 + sigma2) / 2
        radius12 = (sigma1 - sigma2) / 2
        center23 = (sigma2 + sigma3) / 2
        radius23 = (sigma2 - sigma3) / 2
        center13 = (sigma1 + sigma3) / 2
        radius13 = (sigma1 - sigma3) / 2

        circ12_x, circ12_y = generate_circle(center12, radius12)
        circ23_x, circ23_y = generate_circle(center23, radius23)
        circ13_x, circ13_y = generate_circle(center13, radius13)

        fig_mohr = go.Figure()
        fig_mohr.add_trace(go.Scatter(x=circ13_x, y=circ13_y, fill="toself", fillcolor='lightblue', line_color='blue', name='Círculo σ₁-σ₃'))
        fig_mohr.add_trace(go.Scatter(x=circ12_x, y=circ12_y, fill="toself", fillcolor='white', line_color='red', name='Círculo σ₁-σ₂'))
        fig_mohr.add_trace(go.Scatter(x=circ23_x, y=circ23_y, fill="toself", fillcolor='white', line_color='green', name='Círculo σ₂-σ₃'))
        fig_mohr.add_trace(go.Scatter(x=[sigma1, sigma2, sigma3], y=[0, 0, 0], mode='markers+text', marker=dict(color='black', size=10), text=[f"σ₁={sigma1:.1f}", f"σ₂={sigma2:.1f}", f"σ₃={sigma3:.1f}"], textposition="top center", name='Tensões Principais'))
        fig_mohr.add_trace(go.Scatter(x=[center13], y=[tau_max_abs], mode='markers+text', marker=dict(color='purple', size=10, symbol='star'), text=[f"τ_max={tau_max_abs:.1f}"], textposition="middle right", name='Tensão Cisalhante Máxima'))
        fig_mohr.update_layout(title_text='Círculo de Mohr', xaxis_title="Tensão Normal (σ)", yaxis_title="Tensão Cisalhante (τ)", xaxis=dict(scaleanchor="y", scaleratio=1), yaxis = dict(zeroline=True, zerolinewidth=2, zerolinecolor='Black'), showlegend=True, width=700, height=700)
        st.plotly_chart(fig_mohr, use_container_width=True)

        st.markdown("---")
        st.subheader("Visualização 3D do Elemento de Tensão")

        s = 1
        vertices = np.array([
            [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
            [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s]
        ])

        # ###############################################################
        # INÍCIO DA SEÇÃO CORRIGIDA
        # ###############################################################

        # CORREÇÃO: Definir as 12 faces triangulares do cubo
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # face traseira
            [4, 5, 6], [4, 6, 7],  # face frontal
            [0, 1, 5], [0, 5, 4],  # face inferior
            [2, 3, 7], [2, 7, 6],  # face superior
            [0, 3, 7], [0, 7, 4],  # face esquerda
            [1, 2, 6], [1, 6, 5]   # face direita
        ])

        R = direction_cosines
        rotated_vertices = vertices @ R.T

        # CORREÇÃO: A função agora não usa o parâmetro inválido 'd'
        def create_cube_trace(v, f, name, color, opacity=1.0):
            return go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                i=f[:, 0], j=f[:, 1], k=f[:, 2], # Apenas i, j, k para os triângulos
                name=name,
                color=color,
                opacity=opacity,
                flatshading=True
            )
        
        # ###############################################################
        # FIM DA SEÇÃO CORRIGIDA
        # ###############################################################

        trace_original = create_cube_trace(vertices, faces, 'Cubo Original (x,y,z)', 'lightpink', opacity=0.3)
        trace_principal = create_cube_trace(rotated_vertices, faces, 'Cubo Principal (σ₁,σ₂,σ₃)', 'cyan', opacity=0.8)
        
        axis_len = 1.8 * s
        origin = [0,0,0]
        axis_x = go.Scatter3d(x=[origin[0], axis_len], y=[origin[1], origin[1]], z=[origin[2], origin[2]], mode='lines+text', line=dict(color='red', width=5), name='Eixo X', text=['', 'x'])
        axis_y = go.Scatter3d(x=[origin[0], origin[0]], y=[origin[1], axis_len], z=[origin[2], origin[2]], mode='lines+text', line=dict(color='green', width=5), name='Eixo Y', text=['', 'y'])
        axis_z = go.Scatter3d(x=[origin[0], origin[0]], y=[origin[1], origin[1]], z=[origin[2], axis_len], mode='lines+text', line=dict(color='blue', width=5), name='Eixo Z', text=['', 'z'])
        
        v1 = R[:, 0] * axis_len
        v2 = R[:, 1] * axis_len
        v3 = R[:, 2] * axis_len
        axis_s1 = go.Scatter3d(x=[origin[0], v1[0]], y=[origin[1], v1[1]], z=[origin[2], v1[2]], mode='lines+text', line=dict(color='purple', width=5, dash='dash'), name='Eixo σ₁', text=['', 'σ₁'])
        axis_s2 = go.Scatter3d(x=[origin[0], v2[0]], y=[origin[1], v2[1]], z=[origin[2], v2[2]], mode='lines+text', line=dict(color='orange', width=5, dash='dash'), name='Eixo σ₂', text=['', 'σ₂'])
        axis_s3 = go.Scatter3d(x=[origin[0], v3[0]], y=[origin[1], v3[1]], z=[origin[2], v3[2]], mode='lines+text', line=dict(color='magenta', width=5, dash='dash'), name='Eixo σ₃', text=['', 'σ₃'])

        fig_3d = go.Figure(data=[
            trace_original, trace_principal,
            axis_x, axis_y, axis_z,
            axis_s1, axis_s2, axis_s3
        ])

        fig_3d.update_layout(
            title='Orientação do Elemento de Tensão Original e Principal',
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z'),
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        st.info("O cubo ciano representa o elemento na orientação das tensões principais. Note como os eixos tracejados (σ₁, σ₂, σ₃) são os eixos de simetria deste novo cubo, e suas faces são perpendiculares a eles.")

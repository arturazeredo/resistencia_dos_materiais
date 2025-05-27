import streamlit as st
import pandas as pd
import numpy as np
from numpy.linalg import linalg as LA
import plotly.graph_objects as go

st.set_page_config(
    page_icon="⚙️",
    page_title="Resistência dos Materiais",
    layout="wide" # wide window mode
    ) 

# Replace with st.image if needed
# st.image("./images/ufu_logo.png", width=200)

st.title("Estado Triplo de Tensões")
st.markdown("Universidade Federal de Uberlândia")
st.markdown("Desenvolvido por: Artur Azeredo Santos Servian")
st.markdown("Sob orientação de: Prof. Leonardo Campanine Sicchieri")

st.write("Insira as tensões no estado tridimensional:")

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
        step=50
    )

with col2:
    value_12 = st.number_input(
        "τₓᵧ = τᵧₓ (Tensão Tangencial no plano xy)", 
        value=None,
        key="m01", 
        placeholder="τₓᵧ",
        step=50
    )
    st.session_state.matrix[0, 1] = value_12
    st.session_state.matrix[1, 0] = value_12

with col3:
    value_13 = st.number_input(
        "τₓᵣ = τᵣₓ (Tensão Tangencial no plano xz)", 
        value=None,
        key="m02",
        placeholder="τₓᵣ",
        step=50
    )
    st.session_state.matrix[0, 2] = value_13
    st.session_state.matrix[2, 0] = value_13

# Row 2
with col1:
    st.text_input("τᵧₓ = τₓᵧ (Tensão Tangencial no plano yx)", value=None, 
    placeholder="τᵧₓ", 
    disabled=True
    )

with col2:
    st.session_state.matrix[1, 1] = st.number_input(
        "σᵧ (Tensão Normal em y)", 
        value=None,
        key="m11",
        placeholder="σᵧ",
        step=50
    )

with col3:
    value_23 = st.number_input(
        "τᵧᵣ = τᵣᵧ (Tensão Tangencial no plano yz)", 
        value=None,
        key="m12",
        placeholder="τᵧᵣ",
        step=50
    )
    st.session_state.matrix[1, 2] = value_23
    st.session_state.matrix[2, 1] = value_23

# Row 3
with col1:
    st.text_input("τᵣₓ = τₓᵣ (Tensão Tangencial no plano zx)", value=None,
    placeholder="τᵣₓ",
    disabled=True
    )
with col2:
    st.text_input("τᵣᵧ = τᵧᵣ (Tensão Tangencial no plano zy)", value=None,
    placeholder="τᵣᵧ",
    disabled=True
    )

with col3:
    st.session_state.matrix[2, 2] = st.number_input(
        "σᵣ (Tensão Normal em z)", 
        value=None,
        key="m22",
        placeholder="σᵣ",
        step=50
    )
st.markdown("---")

# Function to create a 3D visualization of the cube and principal planes
def create_3d_visualization(direction_cosines, principal_stresses):
    # Create a figure
    fig = go.Figure()
    
    # Define the cube vertices
    L = 1  # Side length of the cube
    vertices = np.array([
        [0, 0, 0], [L, 0, 0], [L, L, 0], [0, L, 0],
        [0, 0, L], [L, 0, L], [L, L, L], [0, L, L]
    ])
    
    # Define cube faces (indices of vertices)
    faces = [
        [0, 1, 2, 3],  # bottom face
        [4, 5, 6, 7],  # top face
        [0, 1, 5, 4],  # front face
        [2, 3, 7, 6],  # back face
        [0, 3, 7, 4],  # left face
        [1, 2, 6, 5]   # right face
    ]
    
    # Add cube faces
    for face in faces:
        i, j, k, l = face
        x = [vertices[i][0], vertices[j][0], vertices[k][0], vertices[l][0], vertices[i][0]]
        y = [vertices[i][1], vertices[j][1], vertices[k][1], vertices[l][1], vertices[i][1]]
        z = [vertices[i][2], vertices[j][2], vertices[k][2], vertices[l][2], vertices[i][2]]
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False
        ))
    
    # Add planes for each principal direction
    colors = ['red', 'green', 'blue']
    center = np.array([0.5, 0.5, 0.5])  # Center of the cube
    
    for i in range(3):
        normal = direction_cosines[:, i]
        
        # Calculate plane equation: ax + by + cz = d
        a, b, c = normal
        d = np.dot(normal, center)
        
        # Create a grid for the plane
        xx, yy = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
        
        # Calculate z from the plane equation
        if abs(c) > 1e-10:  # Avoid division by zero
            zz = (d - a*xx - b*yy) / c
        else:
            # Handle case where plane is perpendicular to z-axis
            zz = np.zeros_like(xx) + 0.5
        
        # Clip the plane to the cube boundaries
        mask = (zz >= 0) & (zz <= 1)
        xx_visible = xx[mask]
        yy_visible = yy[mask]
        zz_visible = zz[mask]
        
        # Add the plane if there are visible points
        if len(xx_visible) > 0:
            fig.add_trace(go.Mesh3d(
                x=xx_visible.flatten(), 
                y=yy_visible.flatten(), 
                z=zz_visible.flatten(),
                opacity=0.5,
                color=colors[i],
                name=f"Principal Plane {i+1} (σ = {principal_stresses[i]:.2f})"
            ))
    
    # Update the layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube',
        ),
        title="Visualização 3D dos Planos Principais",
        height=700,
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    return fig

    
# Button to calculate eigenvalues and eigenvectors
if st.button("Calcular Tensões e Direções"):
    if st.session_state.matrix is None or np.isnan(st.session_state.matrix).any():
        st.warning("Por favor preencha as tensões corretamente.")
    else:
        st.subheader("Matriz Tensor Tensão Gerada")
        st.write(st.session_state.matrix)
        T = st.session_state.matrix
        
        # Calculating eigenvectors and eigenvalues
        T_eigval, T_eigvec = LA.eig(T)
        
        # Reorganizing in descending order
        sorted_idx = np.argsort(T_eigval)[::-1]  # Sort indices in descending order
        T_eigval = T_eigval[sorted_idx]  # Get sorted eigenvalues
        T_eigvec = T_eigvec[:, sorted_idx]  # Reorganize eigenvectors according to eigenvalues
        
        ## Adjusting the sign of eigenvectors
        # Ensures that the first component of each eigenvector is positive
        for i in range(3):
            if T_eigvec[0, i] < 0:
                T_eigvec[:, i] = -T_eigvec[:, i]
        
        # Adjustment of the orientation of the orthonormal basis
        if LA.det(T_eigvec) < 0:
            T_eigvec[:, 2] = -T_eigvec[:, 2]  # Invert the third eigenvector to maintain consistency
        
        # Principal stresses (Eigenvalues)
        principal_stresses = T_eigval
        
        # Direction cosines (Eigenvectors)
        direction_cosines = T_eigvec
        
        # Principal directions in degrees
        principal_directions = np.degrees(np.arccos(direction_cosines))

        results_col1, results_col2 = st.columns(2)
        
        with results_col1:
            st.subheader("Tensões Principais (Autovalores)")
            st.latex(r'''
            \begin{align}
            \sigma_1 &= %.2f \\
            \sigma_2 &= %.2f \\
            \sigma_3 &= %.2f
            \end{align}
            ''' % (principal_stresses[0], principal_stresses[1], principal_stresses[2]))
            
            st.subheader("Cossenos diretores (Autovetores)")
            st.write(direction_cosines)
            
            # Show the principal directions with respect to x, y, z axes
            st.subheader("Direções principais (cossenos)")
            for i in range(3):
                st.write(f"Direção Principal {i+1}: [{direction_cosines[0,i]:.4f}, {direction_cosines[1,i]:.4f}, {direction_cosines[2,i]:.4f}]")
        
        with results_col2:
            st.subheader("Direções Principais (em graus)")
            
            # Create a table to display the directions more clearly
            st.write("Ângulos com as coordenadas dos eixos:")
            dir_data = []
            for i in range(3):
                dir_data.append([
                    f"Direção {i+1}",
                    f"{principal_directions[0,i]:.2f}°",
                    f"{principal_directions[1,i]:.2f}°",
                    f"{principal_directions[2,i]:.2f}°"
                ])
            
            st.table({
                "": [row[0] for row in dir_data],
                "X-axis": [row[1] for row in dir_data],
                "Y-axis": [row[2] for row in dir_data],
                "Z-axis": [row[3] for row in dir_data]
            })
            
            st.info("As tensões principais representam os autovalores do tensor tensão, que são as tensões normais que ocorrem quando o sistema de coordenadas é alinhado com as direções principais.")
        
        # Add 3D visualization
        st.subheader("Visualização 3D dos Planos Principais")
        fig = create_3d_visualization(direction_cosines, principal_stresses)
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional explanation
        st.markdown("""
        ### Interpretação da Visualização:
        
        - **Cubo**: Representa o elemento infinitesimal no material.
        - **Planos coloridos**: Representam os planos principais onde atuam as tensões principais.
            - **Plano vermelho**: Direção principal 1 (σ₁)
            - **Plano verde**: Direção principal 2 (σ₂)
            - **Plano azul**: Direção principal 3 (σ₃)
        
        Os planos são posicionados perpendiculares às direções dos autovetores (cossenos diretores) correspondentes. 
        Cada plano representa uma superfície onde atua apenas tensão normal (sem tensões de cisalhamento).
        """)
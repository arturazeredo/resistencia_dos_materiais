import streamlit as st
import pandas as pd
import numpy as np
from numpy.linalg import linalg as LA

st.set_page_config(
    page_icon="⚙️",
    page_title="Resistência dos Materiais",
    layout="wide" # wide window mode
    ) 
st.logo("./images/ufu_logo.png")

st.title("Estado Triplo de Tensões")
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
        placeholder="σₓ"
    )

with col2:
    value_12 = st.number_input(
        "τₓᵧ = τᵧₓ (Tensão Tangencial no plano xy)", 
        value=None,
        key="m01", 
        placeholder="τₓᵧ"
    )
    st.session_state.matrix[0, 1] = value_12
    st.session_state.matrix[1, 0] = value_12

with col3:
    value_13 = st.number_input(
        "τₓᵣ = τᵣₓ (Tensão Tangencial no plano xz)", 
        value=None,
        key="m02",
        placeholder="τₓᵣ"
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
        placeholder="σᵧ"
    )

with col3:
    value_23 = st.number_input(
        "τᵧᵣ = τᵣᵧ (Tensão Tangencial no plano yz)", 
        value=None,
        key="m12",
        placeholder="τᵧᵣ"
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
        placeholder="σᵣ"
    )
st.markdown("---")
def generate_random_matrix():
    normal_stresses = np.random.uniform(0, 500, 3)
    shear_xy = np.random.uniform(-250, 250)
    shear_xz = np.random.uniform(-250, 250)
    shear_yz = np.random.uniform(-250, 250)
    
    # Create the symmetric matrix
    matrix = np.array([
        [normal_stresses[0], shear_xy, shear_xz],
        [shear_xy, normal_stresses[1], shear_yz],
        [shear_xz, shear_yz, normal_stresses[2]]
    ])
    
    return matrix

# Button to generate a random matrix
#####################################
    
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
        principal_directions = np.degrees(np.arccos(np.abs(direction_cosines)))

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
                f"Direction {i+1}",
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
        
        # Add a visualization section
        #st.subheader("Visualization")
        st.info("As tensões principais representam os autovalores do tensor tensão, que são as tensões normais que ocorrem quando o sistema de coordenadas é alinhado com as direções principais.")

        


    
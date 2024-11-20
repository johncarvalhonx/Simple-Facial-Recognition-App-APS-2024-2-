import cv2
import os
import pickle
import numpy as np
import dlib
import logging
from datetime import datetime

# Caminho para armazenar os dados dos usuários registrados
USER_DATA_PATH = 'user_data.pkl'
# Caminho do arquivo de log
LOG_FILE_PATH = 'authentication_logs.log'

# Função para limpar o console
def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

# Configuração de logging para registrar eventos importantes no sistema
logging.basicConfig(filename=LOG_FILE_PATH, level=logging.INFO, format='%(asctime)s - %(message)s')

# Mapeamento de funções para níveis de segurança
ROLE_SECURITY_LEVELS = {
    "User": 1,
    "Director": 2,
    "Minister of Environment": 3
}

# Informação secreta que apenas o Ministro terá acesso.
reckoning_prediction = """
In the year 2089, scientific models predict that a global environmental event, referred to as the Reckoning, will occur. Mother Nature, after centuries of environmental degradation, is expected to reach a tipping point that could result in catastrophic changes. This information, classified at the highest level, is known only to the government. The intent is to prevent widespread panic and disorder among the general public.

The initial phase is expected to involve significant oceanic changes: a gradual, yet persistent, rise in sea levels that will inundate coastal areas. This rise will not occur as a sudden wave, but rather as a consistent and relentless encroachment. Major metropolitan areas, such as New York, Mumbai, and London, are predicted to become submerged, with their structures left as submerged remnants beneath rising waters.

Scientific analysis suggests that terrestrial ecosystems will also undergo significant transformations. Forest biomes are expected to expand and reclaim regions that were previously urbanized. Vegetation, nourished by the absence of human interference, will begin to penetrate and dismantle built environments, effectively erasing suburban landscapes. This process can be viewed as a natural rebalancing effort, aimed at restoring ecological equilibrium.

Atmospheric conditions are also expected to deteriorate, with projections indicating an increase in extreme weather events. Skies may be characterized by atmospheric instability, with an increase in storm frequency and intensity. Unpredictable lightning activity could severely damage infrastructure, while intense rainfall may lead to flash flooding and soil erosion. These phenomena are considered to be nature's attempt to reestablish balance in response to significant anthropogenic disruption.

In the aftermath, societal changes are expected to be profound. Survivors will likely have to adapt to a new paradigm, one in which the lessons of past ecological mismanagement are acknowledged. Reverence for natural systems may replace the prior focus on technological dominance. Modern technology may become obsolete in many regions, as communities turn to traditional methods of subsistence and survival. There will be a renewed emphasis on living in harmony with natural cycles, utilizing sustainable practices in place of resource-intensive technologies.

Mother Nature's response will be indifferent to human intentions or beliefs. She operates according to the principles of ecological balance and resilience, and the Reckoning will be her method of reclaiming what has always been hers. Humanity will be compelled to adapt, transitioning from a dominant force to a small component within a larger ecological framework—if they are to endure at all.

This document is for authorized personnel only. Dissemination of this information to the general public is strictly prohibited.
"""

# Inicializa o detector de rosto e o codificador de rosto do dlib
face_detector = dlib.get_frontal_face_detector()  # Detector frontal de rostos
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")  # Modelo de reconhecimento facial
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Preditor de marcadores faciais

# Carrega os dados dos usuários existentes
# Isso será usado para autenticar ou adicionar novos usuários
# Retorna um dicionário vazio caso o arquivo não exista
def load_user_data():
    if os.path.exists(USER_DATA_PATH):
        with open(USER_DATA_PATH, 'rb') as file:
            return pickle.load(file)
    return {}

# Salva os dados dos usuários no arquivo especificado
def save_user_data(user_data):
    with open(USER_DATA_PATH, 'wb') as file:
        pickle.dump(user_data, file)

# Registra um novo usuário no sistema
def register_user(name, role):
    user_data = load_user_data()
    
    # Verifica se o usuário já existe, prevenindo registros duplicados
    if name in user_data:
        clear_console()
        title_menu("Ministry of the Environment", "Official Government App")
        print(f"\nUser '{name}' already exists. Try a different name.")
        input("\nPress Enter to continue...")
        clear_console()
        title_menu("Ministry of the Environment", "Official Government App")
        return
    
    # Obtém o nível de segurança do papel do usuário
    security_level = ROLE_SECURITY_LEVELS.get(role)
    if security_level is None:
        print("Invalid role selected.")
        return
    
    # Aquisição: Inicialização da webcam para captura do rosto do usuário
    video_capture = cv2.VideoCapture(0)
    print("Press 'q' to capture and register your face.")
    face_descriptor = None
    
    # Loop para capturar imagens da webcam até que um rosto seja detectado
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture image. Exiting registration.")
            break
        
        # Pré-processamento: Converte a imagem capturada para escala de cinza para melhor desempenho na detecção de rostos
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray_frame)

        # Segmentação: Se rostos forem detectados, desenha um retângulo ao redor deles
        if faces:
            for face in faces:
                x, y, w, h = (face.left(), face.top(), face.width(), face.height())
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, 'Face Detected', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Register - Ministry of the Environment", frame)
        
        # Se 'q' for pressionado, captura o rosto
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if faces:
                # Extração de Características: Extração de características do rosto detectado
                face = faces[0]
                shape = shape_predictor(gray_frame, face)
                face_descriptor = np.array(face_encoder.compute_face_descriptor(frame, shape))
                
                if face_descriptor is not None:
                    break
                else:
                    print("Error: No face descriptor was generated. Please try again.")
            else:
                print("No face detected at the moment of capture. Please try again.")
    
    # Libera a câmera e fecha as janelas de visualização
    video_capture.release()
    cv2.destroyAllWindows()
    
    # Salva o descritor facial e os dados do usuário se a captura for bem-sucedida
    if face_descriptor is not None:
        user_data[name] = {
            "face_descriptor": face_descriptor,
            "security_level": security_level,
            "role": role,
            "registration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_user_data(user_data)
        logging.info(f"User '{name}' has been registered successfully with role '{role}'.")
        print(f"User '{name}' has been registered successfully with role '{role}'.")
    else:
        logging.info(f"Registration failed for user '{name}'.")
        print("Registration failed.")

# Exibe o perfil do usuário com informações de papel e nível de segurança
def view_profile(user_data, name):
    clear_console()
    user = user_data.get(name)
    if user:
        clear_console()
        title_menu("Ministry of the Environment", "Official Government App")
        print("User Profile")
        print("============")
        print(f"Name: {name}")
        print(f"Role: {user['role']}")
        print(f"Security Level: {user['security_level']}")
        print("============")
    else:
        clear_console()
        print("User not found.")

# Exibe um menu personalizado de acordo com o papel do usuário autenticado
def display_menu(name, role):
    clear_console()
    title_menu("Ministry of the Environment", "Official Government App")
    print(f"\nWelcome, {role} {name}!")

    # Opções de menu variam conforme o papel do usuário
    if role == "Regular User":
        print("Menu Options:")
        print("1. View Profile")
        print("2. View Settings")
    
    elif role == "Directors":
        print("Menu Options:")
        print("1. View Profile")
        print("2. View Settings")
        print("3. Manage Department")

    elif role == "Minister of Environment":
        print("Menu Options:")
        print("1. View Profile")
        print("2. View Settings")
        print("3. Manage Department")
        print("4. Secret Information - Private Access")

    # Espera o usuário escolher uma opção
    choice = input("Select an option: ")

    user_data = load_user_data()
    if choice == '1':
        clear_console()
        title_menu("Ministry of the Environment", "Official Government App")
        view_profile(user_data, name)
        input("\nPress Enter to continue...")
        display_menu(name, role)
    elif choice == '2':
        clear_console()
        title_menu("Ministry of the Environment", "Official Government App")
        # Caso o usuário seja o Ministro, ele poderá manipular informações de outros usuários
        if role == "Minister of Environment":
            print("Settings Menu:")
            print("1. Update User")
            print("2. Delete User")
            print("3. List All Users")
            choice = input("Select an option: ")
            if choice == '1':
                clear_console()
                name_to_update = input("Enter the name of the user to update: ")
                if name_to_update in user_data:
                    new_role = input("Enter new role (1. Regular User, 2. Directors, 3. Minister of Environment): ")
                    role_map = {
                        '1': "Regular User",
                        '2': "Directors",
                        '3': "Minister of Environment"
                    }
                    updated_role = role_map.get(new_role)
                    if updated_role:
                        user_data[name_to_update]["role"] = updated_role
                        user_data[name_to_update]["security_level"] = ROLE_SECURITY_LEVELS[updated_role]
                        save_user_data(user_data)
                        clear_console()
                        logging.info(f"User '{name_to_update}' has been updated to role '{updated_role}'.")
                        print(f"User '{name_to_update}' has been updated successfully.")
                    else:
                        print("Invalid role choice.")
                else:
                    print(f"User '{name_to_update}' not found.")
                input("\nPress Enter to continue...")
                display_menu(name, role)
            elif choice == '2':
                name_to_delete = input("Enter the name of the user to delete: ")
                if name_to_delete in user_data:
                    if name_to_delete == name:
                        print("You cannot delete yourself.")
                    else:
                        del user_data[name_to_delete]
                        save_user_data(user_data)
                        logging.info(f"User '{name_to_delete}' has been deleted.")
                        print(f"User '{name_to_delete}' has been deleted successfully.")
                else:
                    print(f"User '{name_to_delete}' not found.")
                input("\nPress Enter to continue...")
                display_menu(name, role)
            elif choice == '3':
                print("Listing all users:")
                for user, details in user_data.items():
                    print(f"Name: {user}, Role: {details['role']}, Security Level: {details['security_level']}")
                logging.info(f"User '{name}' listed all users.")
                input("\nPress Enter to continue...")
                display_menu(name, role)
            else:
                print("Wrong option, please insert the correct number next time.")
                input("\nPress Enter to continue...")
                display_menu(name, role)
        else:
            print("\nNo settings avaliable in the moment.")
            input("\nPress Enter to continue...")
            display_menu(name, role)
    elif role == "Directors" or role == "Minister of Environment":
        if choice == "3":
            clear_console()
            title_menu("Ministry of the Environment", "Official Government App")
            print("\nThere is no real department to manage. :D")
            input("\nPress Enter to continue...")
            display_menu(name, role)
    if role == "Minister of Environment":
        if choice == '4':
            clear_console()
            print(reckoning_prediction)
            input("\nPress Enter to continue...")
            display_menu(name, role)
    print("Exiting the program.")
    exit()

# Autentica um usuário com base no reconhecimento facial
def authenticate_user():
    user_data = load_user_data()
    if not user_data:
        print("No users registered. Please register first.")
        return
    
    # Aquisição: Inicialização da webcam para captura de imagem
    video_capture = cv2.VideoCapture(0)
    recognized_user = None

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture image.")
            break

        # Pré-processamento: Converte a imagem capturada para escala de cinza para melhor desempenho na detecção de rostos
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray_frame)

        for face in faces:
            shape = shape_predictor(gray_frame, face)
            
            # Extração de Características: Obtenção do descritor facial do usuário atual
            face_descriptor = np.array(face_encoder.compute_face_descriptor(frame, shape))

            for name, data in user_data.items():
                registered_descriptor = data["face_descriptor"]
                user_security_level = data["security_level"]

                # Classificação: Comparação do descritor facial para autenticação
                tolerance = 0.6 - (user_security_level - 1) * 0.1
                distance = np.linalg.norm(registered_descriptor - face_descriptor)
                match = distance < tolerance

                if match:
                    recognized_user = (name, data["role"])
                    break
            if recognized_user:
                break  # Sai do loop externo sobre os descritores faciais

        # Segmentação: Destacando a área do rosto detectado
        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if recognized_user:
                cv2.putText(frame, f"{recognized_user[0]} ({recognized_user[1]})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Login - Ministry of the Environment", frame)

        # Se um usuário for reconhecido, prossegue
        if recognized_user:
            print(f"Authenticated successfully as: {recognized_user[0]} ({recognized_user[1]})")
            logging.info(f"Authentication successful for user: {recognized_user[0]}, Role: {recognized_user[1]}")
            video_capture.release()
            cv2.destroyAllWindows()
            display_menu(recognized_user[0], recognized_user[1])
            break

        # Sai do loop quando 'q' é pressionado
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Authentication failed. No matching face found.")
            logging.info("Authentication failed. No matching face found.")
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Função para exibir o cabeçalho do menu
def title_menu(text, subtitle):
    largura = max(len(text), len(subtitle)) + 10
    print("=" * largura)
    print(f"|| {text.center(largura - 5)}||")
    print("=" * largura)
    print(f"|| {subtitle.center(largura - 5)}||")
    print("=" * largura)

# Função principal para gerenciar a interface do usuário
def main():
    title_menu("Ministry of the Environment", "Official Government App")
    while True:
        print("\nOptions:")
        print("1. Register new user")
        print("2. Login")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ")
        
        if choice == '1':
            clear_console()
            title_menu("Ministry of the Environment", "Official Government App")
            name = input("\nEnter your name: ")
            print("\nSelect your role:")
            print("1. Regular User")
            print("2. Directors")
            print("3. Minister of Environment")
            role_choice = input("\nEnter role (1/2/3): ")
            role = {
                '1': "Regular User",
                '2': "Directors",
                '3': "Minister of Environment"
            }.get(role_choice)
            
            if role:
                register_user(name, role)
            else:
                clear_console()
                title_menu("Ministry of the Environment", "Official Government App")
                print("\nInvalid role choice.")
        
        elif choice == '2':
            authenticate_user()
        
        elif choice == '3':
            logging.info("Program exited by user.")
            print("Exiting the program.")
            break
        else:
            clear_console()
            title_menu("Ministry of the Environment", "Official Government App")
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main()

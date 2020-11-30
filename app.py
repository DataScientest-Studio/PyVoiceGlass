# ######################################################################################################################
# Projet DS - VoiceGlass - Promotion Bootcamp Juillet 2020 - DataScientest.com
#
# Owners:
# Malik ALY MARECAR (https://www.linkedin.com/in/malik-alymarecar-35502859/)
# Lamia BOUGARA (https://www.linkedin.com/in/lamia-bougara-68aaa9124/)
# ######################################################################################################################
import os
import joblib
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from pathlib import Path
# Core pkg
import appfunctions as app
import appsession as session
import streamlit as st
# audio
import pyaudio
import wave

SAMPLE_RATE = 16000    
SINGLE_WORD_CV = 0
SINGLE_WORD_G = 1
MULTIPLE_WORD_CV = 2
PROTOTYPE_URL = 'https://www.youtube.com/watch?v=fMqifP3Ghos'
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 3
WAVE_OUTPUT_FILENAME = "demo/output.wav"


# ######################################################################################################################
# Parameters
# ######################################################################################################################
SINGLE_WORD_DATASET = SINGLE_WORD_G  # Single Word model to be used in the demo section, SINGLE_WORD_CV or SINGLE_WORD_G
ARDUINO_ACTIVATED = 0                # if arduino is plugged using the LCD display
DATASET_FOLDER = "datasets_lite"     # folder with samples from three datasets : 
                                     # 1. Common Voice Single Word (FR): https://commonvoice.mozilla.org/fr/datasets
                                     # 2. Google Single Word: https://www.kaggle.com/c/tensorflow-speech-recognition-challenge    
                                     # 3. Common Voice Multiple Words (from Kaggle.com): https://www.kaggle.com/mozillaorg/common-voice 
# ######################################################################################################################

def main():
    state = session._get_state()
    pages = {
        "Le projet VoiceGlass": page_dashboard,
        "Datasets": page_datasets,
        "Analyse exploratoire": page_eda,
        "Méthodologie": page_methodologie,
        "Modélisation": page_modelisation,
        "Prédiction (démo)": page_demo,
        "Prototype": page_prototype,
        "Conclusion & Perspectives": page_conclusion,
    }
    st.sidebar.title("VoiceGlass")
    st.sidebar.subheader("Menu") 
    page = st.sidebar.radio("", tuple(pages.keys()))
    pages[page](state)
    state.sync()
    st.sidebar.info(
        "Projet DS - Promotion Bootcamp Juillet 2020"
        "\n\n"
        "Participants:"
        "\n\n"
        "Malik ALY MARECAR (https://www.linkedin.com/in/malik-alymarecar-35502859/)"
        "\n\n"
        "Lamia BOUGARA\n(https://www.linkedin.com/in/lamia-bougara-68aaa9124/) "
        )
    
# ######################################################################################################################
# page HOME
# ######################################################################################################################
        
def page_dashboard(state):
    st.title("VoiceGlass")
    st.header("Lunettes connectées pour malentendants")
    st.write("\n\n")  
    img = Image.open("images/voiceglass.png")
    st.image(img, width = 600, caption = "")
    st.write(
    "Les personnes malentendantes souffrent d’un problème auditif et de ce fait, se trouvent dans l’incapacité de communiquer aisément avec autrui."
    "\n\n"
    "Les lunettes connectées <b>VoiceGlass</b> sont dotées de la technologie de reconnaissance vocale avec des algorithmes de deep learning en termes d’intelligence artificielle." 
    "\n\n"
    "Elles permettent de localiser la voix d’un interlocuteur puis d’afficher sur les verres la transcription textuelle en temps réel."
    "\n\n"
    "Si nous parvenons à fabriquer ces lunettes, il est clair que nous pourrons améliorer la vie des malentendants en leur apportant un confort de vie sans égal."
    "\n\n"
    "C’est tout l’objectif de notre projet.", unsafe_allow_html=True)  
    
# ######################################################################################################################
# page Datasets
# ######################################################################################################################
def page_datasets(state):
    st.title("Les datasets")
    st.header("Le dataset Common Voice de Mozilla")
    st.subheader("Les différentes versions")    
    st.write(
    "Le Dataset Common Voice est sous la licence CC0, c'est-à-dire qu'il est disponible intégralement dans le domaine public avec la libre modification des données. Il est constitué de fichiers MP3 avec les textes transcrits dans des fichiers CSV ou bien TSV "
    "\n\n"
    "Il existe plusieurs versions de datasets Common Voice :"
    "\n\n"
     "•	La version standard Common Voice de Mozilla, qui fait 53,75 Go (https://commonvoice.mozilla.org/fr/datasets)"
    "\n\n"
    "Il s’agit de la version complète qui regroupe des phrases enregistrées par la communauté Common Voice. Ce dataset existe bien entendu dans différentes langues et la taille n’est évidemment pas la même selon la langue choisie. Par exemple, la version Common Voice complète en français fait 16,96 Go."
    "\n\n"       
    "\n\n"
    "•	La version Common Voice hébergée sur Kaggle.com, qui fait une taille de 12,93 (https://www.kaggle.com/mozillaorg/common-voice)"
    "\n\n"
    "Il s’agit d’une version allégée du dataset en anglais disponible sur le site de Mozilla. Elle comprend des phrases enregistrées par la communauté Common Voice. Par convention, nous l'appelons <b>Common Voice Multiple Words</b>."
    "\n\n"       
    "\n\n"
    "•	La version française de <b>Common Voice Single Word</b>, qui fait une taille de 6,75 Go (https://commonvoice.mozilla.org/fr/datasets)"
    "\n\n"
    "Il s’agit d’un segment ciblé de mots pour des cas d’utilisation tels que la reconnaissance des chiffres parlés ou bien la détection oui/non.", unsafe_allow_html=True)         
    st.warning("Nous utilisons dans cette analyse, les datasets Common Voice Single Word mais aussi Common Voice Multiple Words. Ce dernier est un bon compromis en termes de volumétrie par rapport à la version standard")              
    st.header("Le dataset Speech Commands de Google")    
    st.write(
    "Cette version qui est hébergée sur le site de Kaggle.com en tant que « TensorFlow Speech Recognition Challenge » fait 2,05 Go (https://www.kaggle.com/c/tensorflow-speech-recognition-challenge)"
    "\n\n"
    "Il s’agit de fichiers au format WAV rangés dans 30 dossiers labellisés. Chaque dossier correspond ainsi le texte transcrit avec un mot simple des fichiers WAV qu’il comporte."
    "\n\n"
     "Par convention, nous appelons ce dataset Google Single Word."
    "\n\n", unsafe_allow_html=True)         

# ######################################################################################################################
# page EDA
# ######################################################################################################################    
def page_eda(state):    
    st.title("Analyse exploratoire des données")
    st.warning("Dans cette section, nous faisons une analyse non exhaustive en portant une attention particulière aux fichiers audio qui sont les données d'entrées et aux textes transcrits qui sont les données cibles. L'analyse exploratoire complète est traité dans notre rapport technique d'évaluation.")
    st.header("Le dataset Common Voice Single Word")    
    st.write("\n\n")
    data_sw_load_state = st.text('Loading data...')
    data_sw = pd.read_csv(DATASET_FOLDER + "/cv-sw.tsv", sep = "\t")
    data_sw_load_state.text("")
    st.dataframe(data_sw.head(10))
    if st.checkbox("Afficher le résumé "):
        st.write(data_sw.describe())
    if st.checkbox("Afficher les dimensions "):
        st.write(data_sw.shape)   
    st.write("<u>Occurences</u>", unsafe_allow_html=True)   
    fig = plt.figure(figsize=(7, 5))
    plt.title("Occurences des mots")
    data_sw['sentence'].value_counts().plot(kind = 'pie')
    st.pyplot(fig)
    st.info("Nous remarquons que les données sont correctement équilibrées")      
    if st.checkbox("Afficher les nombres pour les occurences"):
        st.write(data_sw['sentence'].value_counts())
    st.subheader("Aperçu d'un fichier audio")
    st.write("\n\n")
    st.write("<u>Fichier :</u> common_voice_fr_22157149.mp3 (correspondant à la ligne 4 du dataframe)", unsafe_allow_html=True)    
    st.write("<u>Transcription :</u><i> Firefox</i>", unsafe_allow_html=True)    
    st.write("\n\n")  
    audio_file = open("demo/cv-sw/common_voice_fr_22157149.mp3", "rb").read()
    st.audio(audio_file)
    audio_path = "demo/cv-sw/common_voice_fr_22157149.mp3"
    audio,fe = app.load_audio(audio_path)
    st.write("\n\n")  
    st.write("<u>Signal audio :</u>", unsafe_allow_html=True)   
    img = Image.open("images/firefox_row.png")
    st.image(img, width = 600, caption = "")
    # ##################################################################################################################
    st.header("Le dataset Google Single Word")    
    st.subheader("Aperçu du fichier csv")
    # Single Google
    list_filename = []
    list_label = []
    #
    data_load_state = st.text('Loading data... ')
    train_audio_path = DATASET_FOLDER + '/g-sw'
    train_labels = os.listdir(train_audio_path)
    file = []
    labels = []
    for label in train_labels:
        if label != '_background_noise_':
            data_folder = train_audio_path + '/' + label
            files = glob(os.path.join(data_folder, '*'))
            for f in files:
                if f.endswith('wav'):
                    file.append(f)
                    labels.append(label)
    train = pd.DataFrame({'file':file,'label':labels})
    data_g = train.sample(5)
    st.dataframe(data_g)
    if st.checkbox("Afficher le résumé  "):
        st.write(train.describe())
    if st.checkbox("Afficher les dimensions  "):
        st.write(train.shape)       
    st.write("<u>Occurences</u>", unsafe_allow_html=True)   
    fig = plt.figure(figsize=(7, 5))
    plt.title("Occurences des mots")
    train['label'].value_counts().plot(kind = 'pie')
    st.pyplot(fig)    
    st.info("Là encore, nous remarquons que les données sont correctement équilibrées avce les 30 classes")   
    st.write("\n\n")
    # ##################################################################################################################
    st.header("Le dataset Common Voice Multiple Words")    
    st.subheader("Aperçu du fichier csv")
    st.write("\n\n")
    data_load_state = st.text('Loading data...')
    data = app.load_data(DATASET_FOLDER + "/cv.csv")
    data_load_state.text("")
    st.dataframe(data.head(10))
    if st.checkbox("Afficher le résumé"):
        st.write(data.describe())
    if st.checkbox("Afficher les dimensions"):
        st.write(data.shape)
    st.subheader("Aperçu d'un fichier audio")
    st.write("\n\n")
    st.write("<u>Fichier :</u> sample-000004.mp3 (correspondant à la ligne 4 du dataframe)", unsafe_allow_html=True)    
    st.write("<u>Transcription :</u><i> he moved about invisible but everyone could hear him</i>", unsafe_allow_html=True)    
    st.write("\n\n")  
    audio_file = open("demo/cv/sample-000004.mp3", "rb").read()
    st.audio(audio_file)
    audio_path = "demo/cv/sample-000004.mp3"
    audio,fe = app.load_audio(audio_path)
    st.write("\n\n")  
    st.write("<u>Signal audio :</u>", unsafe_allow_html=True)   
    img = Image.open("images/sample-000004.png")
    st.image(img, width = 600, caption = "")
    st.write("\n\n")  
    st.write("<u>Log Mel Spectrogrammme :</u>", unsafe_allow_html=True)   
    img = Image.open("images/spectrogramme.png")
    st.image(img, width = 600, caption = "")
    st.info("Le spectrogramme permet une meilleure visibilité puisqu’il est une représentation tridimensionnelle du signal audio (intensité x temps x fréquence)")
    
# ######################################################################################################################
# page MODELISATION SPEECHNET
# ######################################################################################################################
def page_modelisation(state):
    st.title("Modélisation")
    st.header("1. Modélisation simple - Single Word")
    st.write("Nous utilisations un modèle CNN à une dimension puisque nous prédisons des mots simples." 
             "\n\n" , unsafe_allow_html=True)    
    st.subheader("Architecture")
    st.write("\n\n")  
    img = Image.open("images/architecture1.png")
    st.image(img, width = 600, caption = "")             
    if st.checkbox("Afficher la description"):    
        st.write("<b>Partie 1 :</b>"
        "\n\n"
        "•	Une couche convolution Conv1D avec 64 filtres de forme (5)"
         "\n\n"
        "•	Une fonction d'activation LeakyReLU()"
         "\n\n"             
        "•	Un Dropout avec 20% d'éléments éliminés"
        "<b>Partie 2 :</b>"
         "\n\n"             
        "•	Une couche convolution Conv1D avec 128 filtres de forme (5)"
         "\n\n"    
        "•	Une fonction d'activation LeakyReLU ()"
         "\n\n"    
        "•	Un Dropout avec 20% d'éléments éliminés"
         "\n\n"    
        "<b>Partie 3 :</b>"
         "\n\n"    
        "•	Une couche convolution Conv1D avec 256 filtres de forme (5)"
         "\n\n"    
        "•	Une fonction d'activation LeakyReLU ()"
        "\n\n"    
        "•	Un Dropout avec 20% d'éléments éliminés."
        "\n\n"    
        "•	Une couche GlobalAveragePooling"
         "\n\n"    
        "<b>Partie 4 (classification) :</b>"
        "\n\n"            
        "•	Une couche Dense de 256 neurones."
        "\n\n"   
        "•	Un Dropout avec 20% d'éléments jetés."
        "\n\n"       
        "•	Une fonction d'activation LeakyReLU()"
        "\n\n"   
        "•	Une couche Dense de 30 neurones avec le dataset Google Single Word ou bien 14 neurones avec le dataset Common Voice Single Word."
        "•	Une fonction d’activation softmax en sortie"
        "\n\n" , unsafe_allow_html=True)    
    st.subheader("Entraînement")
    st.write("Dans cette modélisation, l’entrainement s’est fait sur un batch_size de 64 et 50 epochs."
              "\n\n" , unsafe_allow_html=True)      
    st.subheader("Évaluation ")
    st.write("Avec le modèle entraîné sur le dataset Google Single Word, nous pouvons remarquer un score de 92%, ce qui est assez remarquable."
             "\n\n" , unsafe_allow_html=True)
    if st.checkbox("Afficher la matrice de confusion"): 
        img = Image.open("images/matrice.png")
        st.image(img, width = 600, caption = "")       
    if st.checkbox("Afficher le rapport de classification"): 
        img = Image.open("images/classification_report.png")
        st.image(img, width = 600, caption = "")      
    st.header("2. Modélisation avancée - Multiple Words")  
    st.write(""
            "\n\n" , unsafe_allow_html=True)
    st.warning("Nous utilisations le modèle SpeechNet, qui est une variante du modèle WaveNet pour traiter les mots multiples. Pour la suite, nous utilisons un modèle pré-entraîné de SpeechNet et nous afficher l'architecture")
    st.subheader("Architecture")
    st.write("\n\n")
    img = Image.open("images/speechnet1.png")
    st.image(img, width = 600, caption = "")
    if st.checkbox("Afficher la description "):    
        st.write("L’architecture SpeechNet est définie en 3 parties :"
        "\n\n"
        "<b>Partie 1 :</b> la couche Expand"
        "\n\n"      
        "Cette première couche de convolution permet d'extraire les caractéristiques pertinentes du fichier audio."
        "\n\n"      
        "<b>Partie 2 :</b> le Residual Stack"
        "\n\n"
        "Cette seconde couche est la partie essentielle. Nous allons décrire brièvement cette partie par la suite."
        "\n\n"      
        "<b>Partie 3 :</b> l’Output Logits"
        "\n\n"      
        "Cette troisième couche est la couche de sortie" , unsafe_allow_html=True)
    if st.checkbox("Afficher le Residual Stack "):
        img = Image.open("images/speechnet2.png")
        st.image(img, width = 600, caption = "")        
        st.write("u sein de la Residual Stack, nous pouvons trouver plusieurs Residual Blocks qui sont connectés les uns avec les autres. Ils sont également reliés aux skip connections sont des connections additionnels entre les nœuds du réseau."
        "\n\n"
        "\n\n", unsafe_allow_html=True)        
    if st.checkbox("Afficher le Residual block "):
        img = Image.open("images/speechnet3.png")
        st.image(img, width = 600, caption = "")
        st.write("Il s’agit du bloc où le traitement lourd est effectué."
        "\n\n"
        "Les entrées dans chaque Residual Block passent d’abord par une couche de convolution dilatées 1D puis les fonctions d’activation tanh() et sigmoid() sont appliquées."
        "\n\n"      
        "Le gate permet alors de gérer la quantité de données passé du filtre. Ce fonctionnement est semblable à celui d’un transistor qui est un composant électronique essentiel dans les mémoires dynamiques RAM des ordinateurs. "
        "\n\n"      
        "<b>Avec le gate, lorque nous avons la valeur 1 retournée par la fonction d’activation sigmoid(), le message est de « laisser tout passer » tandis qu’avec la valeur 0, le message est de « ne laissez rien laisser passer»."
        "\n\n"
        "Enfin, le résultat devient l’élement wise-multiply sur lequel la fonction d’activation tanh() est de nouveau appliquée."
        "\n\n", unsafe_allow_html=True)
    
# ######################################################################################################################
# page PREDICTION
# ######################################################################################################################
def page_demo(state):
    st.set_option('deprecation.showfileUploaderEncoding', False)
    options = ["Ajouter un fichier mp3", "Choisir un fichier mp3 sur le serveur - Single Word", "Choisir un fichier mp3 sur le serveur - Multiple Words", "Enregistrer un audio - Single Word"]                                         
    options2 = ["échantillon 1", "échantillon 2", "échantillon 3", "échantillon 4", "échantillon 5"]
    options3 = ["Modèle simple - Single Word", "Modèle avancée - Multiple Words"]                                        
    st.title("Prédiction (Démo)")
    st.header("Choix du fichier")
    state.selectbox = st.selectbox("", options, options.index(state.selectbox) if state.selectbox else 0)
    
    # 1. DRAG & DROP ##################################################################################
    if options.index(state.selectbox) == 0:
        audio = st.file_uploader("", type = ["mp3"])
        if audio is not None:
            audio = AudioSegment.from_mp3(audio)
            audio.export("demo/temp.mp3", format="mp3")
            audio_path = "demo/temp.mp3"
            audio_file = open(audio_path, "rb").read()
            st.write("Fichier audio :", unsafe_allow_html=True) 
            st.audio(audio_file)
            st.header("Choix du modèle")
            choice_model_def = st.radio("", options3)
            choice_model = options3.index(choice_model_def)             
            # st.write(choice_model)
            if choice_model == 0:
                chosen_model = SINGLE_WORD_DATASET
                st.write("Prédiction sur le modéle simple - Single Word")
                st.write("Classes prédites :", unsafe_allow_html=True) 
                if SINGLE_WORD_DATASET == SINGLE_WORD_G:
                    st.info("right, five, zero, cat, yes, six, down, house, sheila, three, off, left, bed, happy, eight, bird, nine, tree, one, no, go, on, stop, seven, dog, four, wow, up, two, marvin")  
                elif SINGLE_WORD_DATASET == SINGLE_WORD_CV:
                    st.info("neuf, hey, oui, Firefox, trois, sept, zéro, non, six, huit, quatre, cinq, un, deux")
            elif choice_model == 1:
                chosen_model = MULTIPLE_WORD_CV
                st.write("Prédiction sur le modéle avancée - Multiple Words")
            if st.button("Lancer la prédiction "):
                text = app.predict(chosen_model, audio_path)
                if chosen_model == SINGLE_WORD_DATASET:
                    st.success(text)
                elif chosen_model == MULTIPLE_WORD_CV:
                    st.success(text[0])
                if ARDUINO_ACTIVATED == 1:
                    app.sendtext_arduino(text[0])
                    st.info("VoiceGlass OK")
            
    # 2. FICHIER SERVEUR SINGLE WORD ################################################################################ 
    elif options.index(state.selectbox) == 1:
        st.write("Classes prédites :", unsafe_allow_html=True) 
        if SINGLE_WORD_DATASET == SINGLE_WORD_G:
            st.info("right, five, zero, cat, yes, six, down, house, sheila, three, off, left, bed, happy, eight, bird, nine, tree, one, no, go, on, stop, seven, dog, four, wow, up, two, marvin")  
        elif SINGLE_WORD_DATASET == SINGLE_WORD_CV:
            st.info("neuf, hey, oui, Firefox, trois, sept, zéro, non, six, huit, quatre, cinq, un, deux")
        st.write("Echantillons :")
        state.radio = st.radio("", options2, options2.index(state.radio) if state.radio else 0)
        choice = options2.index(state.radio) 
        if choice == 0:
            if SINGLE_WORD_DATASET == SINGLE_WORD_G:
                audio_path = "demo/g-sw/8781f4c1_nohash_0.wav"
                audio_file = open(audio_path, "rb").read()
                st.write("<u>Fichier :</u> 8781f4c1_nohash_0.wav", unsafe_allow_html=True) 
                st.write("<u>Transcription :</u><i> six</i>", unsafe_allow_html=True)    
                st.write("\n\n") # (data, format='audio/wav', start_time=0)
                st.audio(audio_file, format='audio/wav', start_time=0)
            elif SINGLE_WORD_DATASET == SINGLE_WORD_CV:
                audio_path = "demo/cv-sw/common_voice_fr_22221284.mp3"
                audio_file = open(audio_path, "rb").read()
                st.write("<u>Fichier :</u> common_voice_fr_22221284.mp3", unsafe_allow_html=True) 
                st.write("<u>Transcription :</u><i> Hey</i>", unsafe_allow_html=True)    
                st.write("\n\n") # (data, format='audio/wav', start_time=0)
                st.audio(audio_file)
        elif choice == 1:
            if SINGLE_WORD_DATASET == SINGLE_WORD_G:
                audio_path = "demo/g-sw/7d5f87c8_nohash_0.wav"
                audio_file = open(audio_path, "rb").read()
                st.write("<u>Fichier :</u> 7d5f87c8_nohash_0.wav", unsafe_allow_html=True) 
                st.write("<u>Transcription :</u><i> house</i>", unsafe_allow_html=True)    
                st.write("\n\n") 
                st.audio(audio_file, format='audio/wav')    
            elif SINGLE_WORD_DATASET == SINGLE_WORD_CV:
                audio_path = "demo/cv-sw/common_voice_fr_21925937.mp3"
                audio_file = open(audio_path, "rb").read()
                st.write("<u>Fichier :</u> common_voice_fr_21925937.mp3", unsafe_allow_html=True) 
                st.write("<u>Transcription :</u><i> deux</i>", unsafe_allow_html=True)    
                st.write("\n\n") # (data, format='audio/wav', start_time=0)
                st.audio(audio_file)           
        elif choice == 2:
            if SINGLE_WORD_DATASET == SINGLE_WORD_G:
                audio_path = "demo/g-sw/ed0720f1_nohash_0.wav"
                audio_file = open(audio_path, "rb").read()
                st.write("<u>Fichier :</u> ed0720f1_nohash_0.wav", unsafe_allow_html=True) 
                st.write("<u>Transcription :</u><i> left</i>", unsafe_allow_html=True)    
                st.write("\n\n") 
                st.audio(audio_file)
            elif SINGLE_WORD_DATASET == SINGLE_WORD_CV:
                audio_path = "demo/cv-sw/common_voice_fr_22041369.mp3"
                audio_file = open(audio_path, "rb").read()
                st.write("<u>Fichier :</u> common_voice_fr_22041369.mp3", unsafe_allow_html=True) 
                st.write("<u>Transcription :</u><i> troix</i>", unsafe_allow_html=True)    
                st.write("\n\n") # (data, format='audio/wav', start_time=0)
                st.audio(audio_file)  
        elif choice == 3:
            if SINGLE_WORD_DATASET == SINGLE_WORD_G:
                audio_path = "demo/g-sw/da584bc0_nohash_0.wav"
                audio_file = open(audio_path, "rb").read()
                st.write("<u>Fichier :</u> da584bc0_nohash_0.wav", unsafe_allow_html=True) 
                st.write("<u>Transcription :</u><i> tree</i>", unsafe_allow_html=True)    
                st.write("\n\n") 
                st.audio(audio_file)
            elif SINGLE_WORD_DATASET == SINGLE_WORD_CV:
                audio_path = "demo/cv-sw/common_voice_fr_22163721.mp3"
                audio_file = open(audio_path, "rb").read()
                st.write("<u>Fichier :</u> common_voice_fr_22163721.mp3", unsafe_allow_html=True) 
                st.write("<u>Transcription :</u><i> oui</i>", unsafe_allow_html=True)    
                st.write("\n\n") # (data, format='audio/wav', start_time=0)
                st.audio(audio_file)                  
        elif choice == 4:
            if SINGLE_WORD_DATASET == SINGLE_WORD_G:
                audio_path = "demo/g-sw/1ed557b9_nohash_0.wav"
                audio_file = open(audio_path, "rb").read()
                st.write("<u>Fichier :</u> 1ed557b9_nohash_0.wav", unsafe_allow_html=True) 
                st.write("<u>Transcription :</u><i> two</i>", unsafe_allow_html=True)    
                st.write("\n\n") 
                st.audio(audio_file)
            elif SINGLE_WORD_DATASET == SINGLE_WORD_CV:
                audio_path = "demo/cv-sw/common_voice_fr_21959597.mp3"
                audio_file = open(audio_path, "rb").read()
                st.write("<u>Fichier :</u> common_voice_fr_21959597.mp3", unsafe_allow_html=True) 
                st.write("<u>Transcription :</u><i> Firefox</i>", unsafe_allow_html=True)    
                st.write("\n\n") # (data, format='audio/wav', start_time=0)    
                st.audio(audio_file)
        if st.button("Lancer la prédiction"):
            choice_model = SINGLE_WORD_DATASET
            st.write("Prédiction sur le modèle simple - Single Word")
            text = app.predict(choice_model, audio_path)
            st.success(text)  # OK
            if ARDUINO_ACTIVATED == 1:
                app.sendtext_arduino(text[0]) # OK
                st.info("VoiceGlass OK")
                
    # 3. FICHIER SERVEUR MULTIPLE WORDS ################################################################################  
    elif options.index(state.selectbox) == 2:
        st.write("Echantillons :")
        state.radio = st.radio("", options2, options2.index(state.radio) if state.radio else 0)
        choice = options2.index(state.radio)
        if choice == 0:
            audio_path = "demo/cv/sample-003229.mp3"
            audio_file = open(audio_path, "rb").read()
            st.write("<u>Fichier :</u> sample-003229.mp3", unsafe_allow_html=True) 
            st.write("<u>Transcription :</u><i> the boy wanted to believe that his friend had simply become separated from him by accident</i>", unsafe_allow_html=True)    
            st.write("\n\n") 
            st.audio(audio_file) 
        elif choice == 1:
            audio_path = "demo/cv/sample-001813.mp3"
            audio_file = open(audio_path, "rb").read()
            st.write("<u>Fichier :</u> sample-001813.mp3", unsafe_allow_html=True) 
            st.write("<u>Transcription :</u><i> we don't know what's in the confounded thing do you</i>", unsafe_allow_html=True)    
            st.write("\n\n") 
            st.audio(audio_file)
        elif choice == 2:
            audio_path = "demo/cv/sample-001405.mp3"
            audio_file = open(audio_path, "rb").read()
            st.write("<u>Fichier :</u> sample-001405.mp3", unsafe_allow_html=True) 
            st.write("<u>Transcription :</u><i> I heard a faint movement under my feet</i>", unsafe_allow_html=True)    
            st.write("\n\n") 
            st.audio(audio_file) 
        elif choice == 3:
            audio_path = "demo/cv/sample-000223.mp3"
            audio_file = open(audio_path, "rb").read()
            st.write("<u>Fichier :</u> sample-000223.mp3", unsafe_allow_html=True) 
            st.write("<u>Transcription :</u><i> as he looked at the stones he felt relieved for some reason</i>", unsafe_allow_html=True)    
            st.write("\n\n") 
            st.audio(audio_file) 
        elif choice == 4:
            audio_path = "demo/cv/sample-003594.mp3"
            audio_file = open(audio_path, "rb").read()
            st.write("<u>Fichier :</u> sample-003594.mp3", unsafe_allow_html=True) 
            st.write("<u>Transcription :</u><i> I'm going to them</i>", unsafe_allow_html=True)    
            st.write("\n\n") 
            st.audio(audio_file) 
        #
        if st.button("Lancer la prédiction"):
            st.write("Prédiction sur le modèle avancée - Multiple Word")
            text = app.predict(MULTIPLE_WORD_CV, audio_path)
            st.success(text[0])  # OK
            if ARDUINO_ACTIVATED == 1:
                app.sendtext_arduino(text[0])  # OK
                st.info("VoiceGlass OK")
            
    # 4. FICHIER RECORD ################################################################################ 
    elif options.index(state.selectbox) == 3:
        st.write("Classes prédites :", unsafe_allow_html=True) 
        st.info("right, five, zero, cat, yes, six, down, house, sheila, three, off, left, bed, happy, eight, bird, nine, tree, one, no, go, on, stop, seven, dog, four, wow, up, two, marvin")  
        if st.button("Lancer l'enregistrement et la prédiction - durée maximum 3 secondes"):
             with st.spinner('Enregistrement et Traitement en cours...'):
                p = pyaudio.PyAudio()
                stream = p.open(format=FORMAT,channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
                frames = []
                for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                    data = stream.read(CHUNK)
                    frames.append(data)
                stream.stop_stream()
                stream.close()
                p.terminate()
                wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
                audio_path =WAVE_OUTPUT_FILENAME
                st.audio(audio_path)
                st.write("Prédiction sur le modèle simple - Single Word")
                text = app.predict(SINGLE_WORD_G, audio_path)
                st.success(text)
                if ARDUINO_ACTIVATED == 1:
                    app.sendtext_arduino(text[0])
                    st.info("VoiceGlass OK")

# ######################################################################################################################
# page CLASSIFICATION
# ######################################################################################################################
def page_methodologie(state):
    st.title("Méthodologie")
    st.header("Approche")
    st.write("\n\n")  
    st.write(
    "Notre projet de reconnaissance vocale s’apparente à un problème de Deep Learning avec des réseaux de neurones, semblable à une classification d’images."
    "\n\n"
    "Grâce aux différents jeux de données que nous disposons, l’entraînement pour nos modèles est supervisé. Ainsi, on va on va apprendre au réseau à reconnaître les différents phonèmes.", unsafe_allow_html=True)  
    st.header("Exploitation des fichiers audio")
    st.write("\n\n")  
    st.write(
    "Il est nécéssaire d'effectuer les étapes suivantes de preprocessing afin d'assurer l'exploitation des fichiers audio pour les modèles de Deep Learning :"
    "\n\n"
    "1. La normalisation des fichiers avec <a href=""https://pypi.org/project/pynormalize/"" target=""_blank"">pynormalize</a>"
    "\n\n"    
    "2. Le ré-échantillonage des fichiers audio à 16 kHz"
    "\n\n"    
    "3. Le dimensionnement des signaux"
    "\n\n"
    "4. L'application du Log Mel spectrogramme aux données brut traitées", unsafe_allow_html=True)  
    st.warning("L’intérêt d’utiliser le Log Mel spectrogramme est de réduire considérablement le temps d'apprentissage des modèles")


# ######################################################################################################################
# page PROTOYPE
# ######################################################################################################################    
def page_prototype(state):
    st.title("VoiceGlass")
    st.header("Prototype de Lunettes connectées")
    st.write("\n\n")  
    st.video(PROTOTYPE_URL, start_time=1)
    st.write("\n\n")  
    st.subheader("Description")    
    st.write(    
    "Le prototype a été réalisé avec une carte arduino uno sur laquelle nous utilisons un écran LCD que nous plaçons devant les lunettes."
    "\n\n"
    "Le principe est le suivant : "
    "\n\n"
    "La carte reste constamment à l’écoute du texte prédit par la WebApp via une communication série effectué avec python via la librairie <a href=""https://pypi.org/project/pyserial/"" target=""_blank"">pyserial</a>."
    "\n\n", unsafe_allow_html=True)  
    img = Image.open("images/demo.png")
    st.image(img, width = 600, caption = "")
    st.write("Schéma du montage : ")
    st.write("\n\n")  
    #st.header("schéma du montage arduino")    
    img = Image.open("images/arduino_montage.png")
    st.image(img, width = 600, caption = "")
    st.write(            
    "Pour améliorer le prototype, vous prévoyons, d’utiliser à la place de l’écran LCD, d’utiliser un écran OLED transparent de <a href=""https://www.sparkfun.com/"" target=""_blank"">sparkfun</a>. Avec ce composant, les lunettes seraient plus réalistes."
    "\n\n"
    "Nous avions ce composant en notre possession, mais il nous a manqué un connecteur essentiel pour le connecter avec la carte Arduino.", unsafe_allow_html=True)  
    img = Image.open("images/oled.png")
    st.image(img, width = 300, caption = "")
     
# ######################################################################################################################
# page CONCLUSION
# ######################################################################################################################
def page_conclusion(state):
    st.title("Conclusion & Perspectives")
    st.header("Bilan")
    st.write("La mission du projet, qui est d’apporter un confort de vie aux personnes malentendantes, a sollicité en nous le sens de l’empathie. Animés par le désir d’améliorer la vie d’autrui et passionnés par la Data Science, nous avons été suffisamment motivés pour aller jusqu’au bout du projet"
    "\n\n"
    "Nous sommes passés par plusieurs étapes, allant de l’exploration du dataset, à sa visualisation à l’implémentation d’une solution de reconnaissance vocale dans une WebApp et même jusqu’à la réalisation d’un prototype fonctionnel.", unsafe_allow_html=True)  
    st.header("Pistes d’améliorations")
    st.subheader("Modélisation avec RNN et approche NLP")            
    st.write("Avec ce projet, nous avons pu effectuer tout d’abord une modélisation simple avec les réseaux de convolutions 1D puis nous avons effectué une seconde modélisation plus complexe avec le modèle SpeechNet. Cependant, nous avons utilisé un modèle pré-entraîné faute de pouvoir effectuer l’entraînement en question sur nos machines configurées seulement en calcul TPU."
     "\n\n"
    "Pour aller plus loin, nous souhaiterions effectuer une énième modélisation avec les réseaux récurrents RNN avec un entraînement, cette fois-ci, effectué en GPU. A l’issu de cet entraînement, nous pourrions alors effectuer une approche supplémentaire avec le Natural Language Processing afin de retranscrire un texte sans faute, proche du langage humain.", unsafe_allow_html=True)               
    st.subheader("Modélisation avec RNN et approche NLP")     
    st.write("Avec ce projet, nous avons réalisé un prototype de lunettes connectées. Cependant, ce prototype, même s’il est fonctionnel, repose sur le modèle SpeechNet. En soi, il n’est pas très performant."
    "\n\n"
    "Dans l’absolu, nous aurions voulu concevoir un prototype beaucoup plus performant et capable de retranscrire le texte à travers une technologie de reconnaissance vocale de pointe."
    "\n\n"
    "Si tel était le cas, nous aurions aimé que notre prototype prenne sa place sur le marché des lunettes connectées puisque le projet <b>VoiceGlass</b> répond bel et bien à une problématique réelle et concrète de l’industrie.", unsafe_allow_html=True)  
     
# ######################################################################################################################


# ######################################################################################################################
        
if __name__ == '__main__':
    main()
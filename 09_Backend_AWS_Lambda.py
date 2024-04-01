import boto3
import json
import csv
from io import StringIO
import re
import periodictable
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText



def lambda_handler(event, context):
    s3 = boto3.client('s3')
    bucket_name = 'materials-dataset'
    object_key = 'DATA_RAW_29222.csv'

    # Obtiene el objeto S3
    response = s3.get_object(Bucket=bucket_name, Key=object_key)
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

    headers = {
        'Access-Control-Allow-Origin': 'https://bandgap-prediction.s3.amazonaws.com',  
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
    }

    if event['requestContext']['http']['method'] == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': headers,
            'body': 'OPTIONS request'
        }
    
    # Convierte el cuerpo del evento de JSON a un diccionario de Python
    body = json.loads(event['body'])

    if status == 200:
        # Lee el archivo CSV desde S3
        csv_content = response['Body'].read().decode('utf-8')
        
        data_dic = []

        reader = csv.DictReader(StringIO(csv_content))
        
        for row in reader:
            data_dic.append(row)

        mapeo = {
            'Cubic': 1,
            'Tetragonal': 2,
            'Hexagonal': 3,
            'Trigonal': 4,
            'Orthorhombic': 5,
            'Monoclinic': 6,
            'Triclinic': 7
        }

        # Aplica el mapeo a la clave 'crystal_symmetry' en cada diccionario de la lista
        for diccionario in data_dic:
            if diccionario['crystal_symmetry'] in mapeo:
                diccionario['crystal_symmetry'] = mapeo[diccionario['crystal_symmetry']]

        # Lista de todos los elementos posibles
        elementos_posibles = ['Ho', 'Ti', 'Lu', 'Hf', 'Cu', 'Ga', 'Ba', 'Te', 'Au', 'Ce', 'Pa', 'Br', 'Gd', 'Nb', 'Dy', 'Os', 'Nd',
                            'Co', 'Zn', 'Ni', 'V', 'Cd', 'Hg', 'La', 'Cs', 'B', 'Ag', 'Ru', 'Ar', 'Tb', 'Y', 'Sn', 'Fe', 'Yb', 'P',
                            'Sb', 'Er', 'Ir', 'Th', 'Si', 'O', 'F', 'Na', 'Pr', 'N', 'Mo', 'Se', 'Kr', 'Tl', 'Mg', 'Rb', 'Ne', 'Li',
                            'W', 'Eu', 'Al', 'Pm', 'Sr', 'Rh', 'As', 'Pt', 'Cr', 'Pd', 'Tc', 'I', 'Sm', 'Re', 'Ge', 'Cl', 'S', 'U',
                            'H', 'Tm', 'In', 'K', 'Bi', 'Mn', 'Ca', 'Sc', 'Pu', 'Ac', 'Be', 'Xe', 'C', 'He', 'Zr', 'Np', 'Ta', 'Pb']

        # Funciones para extraer elementos y calcular el peso molecular
        def extract_elements(formula):
            pattern = r'([A-Z][a-z]*)(\d*)'
            matches = re.findall(pattern, formula)
            elements = {match[0]: int(match[1]) if match[1] else 1 for match in matches}
            return elements

        def calculate_molecular_weight(elements):
            weight = 0
            for element, count in elements.items():
                weight += getattr(periodictable, element).mass * count
            return weight

        # Inicializar campos para cada elemento en data_dic
        for registro in data_dic:
            for elemento in elementos_posibles:
                count_key = f'{elemento}_count'
                mass_ratio_key = f'{elemento}_mass_ratio'
                registro[count_key] = 0
                registro[mass_ratio_key] = 0

            # Procesar la fórmula de cada registro
            elementos = extract_elements(registro['formula_pretty'])
            peso_molecular = calculate_molecular_weight(elementos)
            registro['molecular_weight'] = peso_molecular

            # Actualizar conteos y proporciones de masa solo para elementos presentes
            for elemento, cantidad in elementos.items():
                count_key = f'{elemento}_count'
                mass_ratio_key = f'{elemento}_mass_ratio'
                registro[count_key] = cantidad
                registro[mass_ratio_key] = (getattr(periodictable, elemento).mass * cantidad) / peso_molecular

        for registro in data_dic:
            # Eliminar 'formula_pretty'
            if 'formula_pretty' in registro:
                del registro['formula_pretty']

            # Convertir 'is_magnetic' y 'is_stable' de cadenas a enteros (True/False a 1/0)
            registro['is_magnetic'] = int(registro['is_magnetic'] == 'True' or registro['is_magnetic'] is True)
            registro['is_stable'] = int(registro['is_stable'] == 'True' or registro['is_stable'] is True)


        data_dic1 = [registro.copy() for registro in data_dic]

        for registro in data_dic:
            registro['band_gap'] = 1 if float(registro['band_gap']) != 0 else 0

        selected_features = ['nsites', 'nelements', 'volume', 'density', 'density_atomic', 'crystal_symmetry', 'symmetry_number', 'sides_abc', 'angles_abc', 'uncorrected_energy_per_atom', 'energy_per_atom', 'formation_energy_per_atom', 'energy_above_hull', 'is_stable', 'efermi', 'is_magnetic', 'total_magnetization', 'num_magnetic_sites', 'molecular_weight', 'Ho_count', 'Ho_mass_ratio', 'Ti_count', 'Ti_mass_ratio', 'Lu_count', 'Lu_mass_ratio', 'Hf_count', 'Hf_mass_ratio', 'Cu_count', 'Cu_mass_ratio', 'Ga_count', 'Ga_mass_ratio', 'Ba_count', 'Ba_mass_ratio', 'Te_count', 'Te_mass_ratio', 'Au_count', 'Au_mass_ratio', 'Ce_count', 'Ce_mass_ratio', 'Pa_count', 'Pa_mass_ratio', 'Br_count', 'Br_mass_ratio', 'Gd_count', 'Gd_mass_ratio', 'Nb_count', 'Nb_mass_ratio', 'Dy_count', 'Dy_mass_ratio', 'Os_count', 'Os_mass_ratio', 'Nd_count', 'Nd_mass_ratio', 'Co_count', 'Co_mass_ratio', 'Zn_count', 'Zn_mass_ratio', 'Ni_count', 'Ni_mass_ratio', 'V_count', 'V_mass_ratio', 'Cd_count', 'Cd_mass_ratio', 'Hg_count', 'Hg_mass_ratio', 'La_count', 'La_mass_ratio', 'Cs_count', 'Cs_mass_ratio', 'B_count', 'B_mass_ratio', 'Ag_count', 'Ag_mass_ratio', 'Ru_count', 'Ru_mass_ratio', 'Ar_count', 'Ar_mass_ratio', 'Tb_count', 'Tb_mass_ratio', 'Y_count', 'Y_mass_ratio', 'Sn_count', 'Sn_mass_ratio', 'Fe_count', 'Fe_mass_ratio', 'Yb_count', 'Yb_mass_ratio', 'P_count', 'P_mass_ratio', 'Sb_count', 'Sb_mass_ratio', 'Er_count', 'Er_mass_ratio', 'Ir_count', 'Ir_mass_ratio', 'Th_count', 'Th_mass_ratio', 'Si_count', 'Si_mass_ratio', 'O_count', 'O_mass_ratio', 'F_count', 'F_mass_ratio', 'Na_count', 'Na_mass_ratio', 'Pr_count', 'Pr_mass_ratio', 'N_count', 'N_mass_ratio', 'Mo_count', 'Mo_mass_ratio', 'Se_count', 'Se_mass_ratio', 'Kr_count', 'Kr_mass_ratio', 'Tl_count', 'Tl_mass_ratio', 'Mg_count', 'Mg_mass_ratio', 'Rb_count', 'Rb_mass_ratio', 'Ne_count', 'Ne_mass_ratio', 'Li_count', 'Li_mass_ratio', 'W_count', 'W_mass_ratio', 'Eu_count', 'Eu_mass_ratio', 'Al_count', 'Al_mass_ratio', 'Pm_count', 'Pm_mass_ratio', 'Sr_count', 'Sr_mass_ratio', 'Rh_count', 'Rh_mass_ratio', 'As_count', 'As_mass_ratio', 'Pt_count', 'Pt_mass_ratio', 'Cr_count', 'Cr_mass_ratio', 'Pd_count', 'Pd_mass_ratio', 'Tc_count', 'Tc_mass_ratio', 'I_count', 'I_mass_ratio', 'Sm_count', 'Sm_mass_ratio', 'Re_count', 'Re_mass_ratio', 'Ge_count', 'Ge_mass_ratio', 'Cl_count', 'Cl_mass_ratio', 'S_count', 'S_mass_ratio', 'U_count', 'U_mass_ratio', 'H_count', 'H_mass_ratio', 'Tm_count', 'Tm_mass_ratio', 'In_count', 'In_mass_ratio', 'K_count', 'K_mass_ratio', 'Bi_count', 'Bi_mass_ratio', 'Mn_count', 'Mn_mass_ratio', 'Ca_count', 'Ca_mass_ratio', 'Sc_count', 'Sc_mass_ratio', 'Pu_count', 'Pu_mass_ratio', 'Ac_count', 'Ac_mass_ratio', 'Be_count', 'Be_mass_ratio', 'Xe_count', 'Xe_mass_ratio', 'C_count', 'C_mass_ratio', 'He_count', 'He_mass_ratio', 'Zr_count', 'Zr_mass_ratio', 'Np_count', 'Np_mass_ratio', 'Ta_count', 'Ta_mass_ratio', 'Pb_count', 'Pb_mass_ratio']

        X_list = []
        y_list = []

        for registro in data_dic:
            # Añadir las características seleccionadas a X_list y la etiqueta a y_list
            X_list.append([registro[feature] for feature in selected_features])
            y_list.append(registro["band_gap"])

        # Convertir listas a arrays de Numpy
        X_train = np.array(X_list, dtype=float)
        y_train = np.array(y_list, dtype=float)

        # Inicializar y aplicar MinMaxScaler
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)

        data_dic_test = {
            "formula_pretty": str(body["formula_pretty"]),
            "nsites": int(body["nsites"]),
            "nelements": int(body["nelements"]),
            "volume": float(body["volume"]),
            "density": float(body["density"]),
            "density_atomic": float(body["density_atomic"]),
            "crystal_symmetry": str(body["crystal_symmetry"]),
            "symmetry_number": int(body["symmetry_number"]),
            "sides_abc": float(body["sides_abc"]),
            "angles_abc": float(body["angles_abc"]),
            "uncorrected_energy_per_atom": float(body["uncorrected_energy_per_atom"]),
            "energy_per_atom": float(body["energy_per_atom"]),
            "formation_energy_per_atom": float(body["formation_energy_per_atom"]),
            "energy_above_hull": float(body["energy_above_hull"]),
            "is_stable": bool(body["is_stable"]),
            "efermi": float(body["efermi"]),
            "is_magnetic": bool(body["is_magnetic"]),
            "total_magnetization": float(body["total_magnetization"]),
            "num_magnetic_sites": float(body["num_magnetic_sites"])
        }

        if data_dic_test['crystal_symmetry'] in mapeo:
            data_dic_test['crystal_symmetry'] = mapeo[data_dic_test['crystal_symmetry']]

        for elemento in elementos_posibles:
            count_key = f'{elemento}_count'
            mass_ratio_key = f'{elemento}_mass_ratio'
            data_dic_test[count_key] = 0
            data_dic_test[mass_ratio_key] = 0

        elementos = extract_elements(data_dic_test['formula_pretty'])
        peso_molecular = calculate_molecular_weight(elementos)
        data_dic_test['molecular_weight'] = peso_molecular

        for elemento, cantidad in elementos.items():
            count_key = f'{elemento}_count'
            mass_ratio_key = f'{elemento}_mass_ratio'
            data_dic_test[count_key] = cantidad
            data_dic_test[mass_ratio_key] = (getattr(periodictable, elemento).mass * cantidad) / peso_molecular

        # Eliminar 'formula_pretty'
        if 'formula_pretty' in data_dic_test:
            del data_dic_test['formula_pretty']

        # Convertir 'is_magnetic' y 'is_stable' de cadenas a enteros (True/False a 1/0)
        data_dic_test['is_magnetic'] = int(data_dic_test['is_magnetic'] == 'True' or data_dic_test['is_magnetic'] is True)
        data_dic_test['is_stable'] = int(data_dic_test['is_stable'] == 'True' or data_dic_test['is_stable'] is True)

        lista_data_test = [data_dic_test[feature] for feature in selected_features]

        # Convertir lista_data_test a un array de Numpy y cambiar su forma a (1, n_features)
        X_test = np.array(lista_data_test).reshape(1, -1)

        scaler.fit(X_train)

        # Ahora, usar el mismo escalador para transformar tu instancia de test
        X_test = scaler.transform(X_test)

        #MODELO DE CLASIFICACION
        # Inicializar el clasificador de Random Forest
        best_params={'n_estimators': 681, 'min_samples_split': 4, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': 37, 'bootstrap': False}
        rf = RandomForestClassifier(**best_params,random_state=42)

        # Entrenar el modelo con el conjunto de entrenamiento
        rf.fit(X_train, y_train)

        # Realizar predicciones en el conjunto de prueba
        y_pred = rf.predict(X_test)
        y_pred=y_pred.tolist()
        y_pred=int(y_pred[0])

        #Si clasificacion no es 0 para a modelo de regresion
        if y_pred==0:
            result = {'concatenatedParams': y_pred}
        else:

            data_dic = [registro.copy() for registro in data_dic1]

            X_list = []
            y_list = []

            for registro in data_dic:
                # Añadir las características seleccionadas a X_list y la etiqueta a y_list
                X_list.append([registro[feature] for feature in selected_features])
                y_list.append(registro["band_gap"])

            # Convertir listas a arrays de Numpy
            X_train = np.array(X_list, dtype=float)
            y_train = np.array(y_list, dtype=float)

            # Inicializar y aplicar MinMaxScaler
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)

            #MODELO DE REGRESION 

            gb_parameters ={'learning_rate': 0.05, 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 620, 'subsample': 0.99} 
            gb_model = GradientBoostingRegressor(**gb_parameters, random_state=42)

            # Entrenamiento
            gb_model.fit(X_train, y_train)

            # Predicciones individuales
            y_pred = gb_model.predict(X_test)

            y_pred=y_pred.tolist()
            y_pred=y_pred[0]

            result = {'concatenatedParams': y_pred}

        correo_des=str(body["email"])
        print("Correo destinatario: ", correo_des)
        
        #ENVIO RESPUESTA POR CORREO
        def enviar_correo(destinatario, asunto, mensaje):
            try:
                # Configuración del servidor SMTP de Outlook 
                # Configurar cuenta propia de email de donde enviar los correos automaticos
                servidor_smtp = 'smtp.office365.com'
                puerto_smtp = 587
                usuario_smtp = 'correoXXXX@outlook.com'
                password_smtp = 'passXXX'

                # Configuración del mensaje
                msg = MIMEMultipart()
                msg['From'] = usuario_smtp
                msg['To'] = destinatario
                msg['Subject'] = asunto

                # Agregar el cuerpo del mensaje
                msg.attach(MIMEText(mensaje, 'plain'))

                # Configuración del servidor SMTP y envío del mensaje
                with smtplib.SMTP(servidor_smtp, puerto_smtp) as servidor:
                    servidor.starttls()
                    servidor.login(usuario_smtp, password_smtp)
                    servidor.send_message(msg)

            except Exception as e:
                print(f"Error al enviar correo: {str(e)}")

        
        destinatario = correo_des
        asunto = 'Prediccion Bandgap'
        mensaje = "El valor del Bandgap predicho por el modelo es: "+str(y_pred)

        # Llamada a la función para enviar el correo electrónico
        enviar_correo(destinatario, asunto, mensaje)
        print("Correo Enviado")


        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps(result)
        }
    else:
        return {
            'statusCode': status,
            'headers': headers,
            'body': 'Error accessing S3 object'
        }


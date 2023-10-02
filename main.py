import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

app = FastAPI()

games = pd.read_csv('new_games.csv')
review = pd.read_csv('new_review.csv')
items = pd.read_parquet('new_items.parquet')


@app.get('/')
def read_root():
    return {'message' : 'API para consultar datos de Juegos'}

formato_fecha = r'\d{4}-\d{2}-\d{2}'  # Expresión regular para 'YYYY-MM-DD'
games = games[games['release_date'].str.match(formato_fecha, na=False)]

@app.get('/PlayTimeGenre/{genero}')
def PlayTimeGenre(genero: str):
    # Paso 1: Filtrar juegos por género
    juegos_por_genero = games[games['genres'].str.contains(genero, case=False, na=False)]

    if juegos_por_genero.empty:
        return {"No se encontraron juegos para el género": genero}

    # Paso 2: Realizar un join entre "juegos_por_genero" y "items" para obtener las horas jugadas
    merged_df = juegos_por_genero.merge(items, left_on='id', right_on='item_id', how='inner')

    # Paso 3: Extraer el año de lanzamiento de la fecha y sumar las horas jugadas por año
    merged_df['anio_lanzamiento'] = pd.to_datetime(merged_df['release_date']).dt.year
    horas_por_anio = merged_df.groupby('anio_lanzamiento')['playtime_forever'].sum()

    # Paso 4: Encontrar el año con la mayor cantidad de horas jugadas
    anio_max_horas = horas_por_anio.idxmax()

    # Paso 5: Devolver el resultado como un diccionario JSON
    resultado = {"Año de lanzamiento con más horas jugadas para " + genero: str(anio_max_horas)}
    return resultado



@app.get('/UserForGenre/{genero}')
def UserForGenre(genero: str):
    # Paso 1: Filtrar elementos por género
    elementos_por_genero = items[items['item_id'].isin(games[games['genres'].str.contains(genero, case=False, na=False)]['id'])]

    # Paso 2: Crear un diccionario que mapee el 'user_id' al usuario que más horas ha jugado al género dado
    usuario_mas_horas_genero = elementos_por_genero.groupby('user_id')['playtime_forever'].sum().idxmax()

    # Paso 3: Crear un DataFrame que contenga solo los registros del usuario más activo para el género dado
    usuario_mas_horas_df = elementos_por_genero[elementos_por_genero['user_id'] == usuario_mas_horas_genero]

    # Paso 4: Convertir la columna 'release_date' en un formato de fecha y hora
    games['release_date'] = pd.to_datetime(games['release_date'], errors='coerce')

    # Paso 5: Agrupar por año y sumar las horas jugadas
    horas_por_anio = usuario_mas_horas_df.merge(games, left_on='item_id', right_on='id')
    horas_por_anio = horas_por_anio.groupby(horas_por_anio['release_date'].dt.year)['playtime_forever'].sum().reset_index()

    # Paso 6: Renombrar las columnas del DataFrame resultante
    horas_por_anio.columns = ['Año', 'Horas']

    # Paso 7: Devolver el resultado como un diccionario JSON
    resultado = {
        "Usuario con más horas jugadas para " + genero: usuario_mas_horas_genero,
        "Horas jugadas por año para " + genero: horas_por_anio.to_dict(orient='records')
    }
    return resultado

@app.get('/UsersRecommend/{anio}')
def UsersRecommend(anio: int):
    # Paso 1: Filtrar reseñas para el año dado y condiciones de recomendación
    condiciones_filtrado = (review['recommend'] == True) & (review['sentiment_analysis'].isin([1.0, 2.0])) & (review['posted'].str.startswith(str(anio)))
    reseñas_filtradas = review[condiciones_filtrado]

    # Paso 2: Contar cuántas veces se repite cada 'item_id'
    juegos_recomendados = reseñas_filtradas['item_id'].value_counts().head(3)

    # Paso 3: Obtener los nombres de los juegos correspondientes
    juegos_info = games.loc[games['id'].isin(juegos_recomendados.index), ['id', 'app_name']]
    top_juegos_recomendados = [{"Nombre del Juego": row['app_name'], "Cantidad de Recomendaciones": int(juegos_recomendados[row['id']])} for _, row in juegos_info.iterrows()]

    return JSONResponse(content=top_juegos_recomendados)

@app.get('/UsersNotRecommend/{anio}')
def UsersNotRecommend(anio: int):
    # Paso 1: Filtrar reseñas para el año dado y condiciones de recomendación
    condiciones_filtrado = (review['recommend'] == False) & (review['sentiment_analysis'] == 0.0) & (review['posted'].str.startswith(str(anio)))
    reseñas_filtradas = review[condiciones_filtrado]

    # Paso 2: Contar cuántas veces se repite cada 'item_id'
    juegos_recomendados = reseñas_filtradas['item_id'].value_counts().head(3)

    # Paso 3: Obtener los nombres de los juegos correspondientes
    juegos_info = games.loc[games['id'].isin(juegos_recomendados.index), ['id', 'app_name']]
    top_juegos_recomendados = [{"Nombre del Juego": row['app_name'], "Cantidad de No Recomendaciones": int(juegos_recomendados[row['id']])} for _, row in juegos_info.iterrows()]

    return JSONResponse(content=top_juegos_recomendados)


@app.get('/sentiment_analysis/{year}')
def sentiment_analysis(year: int):
    # Paso 1: Convertir la columna 'posted' al formato de fecha
    review['posted'] = pd.to_datetime(review['posted'], errors='coerce')

    # Paso 2: Filtrar reseñas por año
    reviews_year = review[review['posted'].dt.year == year]

    # Paso 3: Contar la cantidad de registros con análisis de sentimiento
    sentiment_counts = reviews_year['sentiment_analysis'].value_counts().to_dict()

    # Paso 4: Devolver los resultados en el formato deseado
    result = {
        "Negative": sentiment_counts.get(0.0, 0),
        "Neutral": sentiment_counts.get(1.0, 0),
        "Positive": sentiment_counts.get(2.0, 0)
    }

    return result


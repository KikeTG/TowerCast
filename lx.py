import streamlit as st
from datetime import datetime
from io import BytesIO, StringIO
def corregir_sellin():
        import os
        import pandas as pd
        from datetime import datetime, timedelta
        import glob
        import numpy as np
        import os
        import pandas as pd
        import streamlit as st
        from datetime import datetime
        import glob
        import numpy as np
        import os
        import pandas as pd
        import streamlit as st
        from datetime import datetime
        import glob
        import numpy as np
        from datetime import datetime
        from io import BytesIO, StringIO

        # Obtener año y mes actual
        fecha_actual = datetime.now()
        año_actual = fecha_actual.year
        mes_actual = fecha_actual.strftime("%m")
        
        uploaded_file_sellin = st.file_uploader("📤 Sube el archivo Sell In (.xlsx)", type="xlsx")
        
        if uploaded_file_sellin is None:
            st.warning("⚠️ Esperando archivo Sell In...")
            st.stop()
        
        try:
            sellin = pd.read_excel(uploaded_file_sellin, sheet_name=0, engine="openpyxl")
            st.success("✅ Archivo Sell In leído correctamente.")
            st.write("Vista previa del archivo:")
            st.dataframe(sellin.head())
        except Exception as e:
            st.error(f"❌ Error al leer el archivo Sell In: {e}")
            st.stop()


        # Procesamiento del archivo Sell In
        sellin.rename(columns={'Nuevo Canal': 'Canal 2'}, inplace=True)
        sellin = sellin[['Ce.', 'Material', 'Texto breve de material', 'Fe.contab.', 'Cliente',
                        'Ult.Eslabón', 'Canal', 'Tipo', 'Forecast AFM', 'Venta UMB',
                        'Semana ISO', 'Canal 2', 'ID']]
        df_sellin = sellin
        df_sellin.rename(columns={'Canal 2': 'Nuevo Canal'}, inplace=True)
        df_sellin_filtered = df_sellin

        # Agrupación y filtrado
        td_sellin = df_sellin_filtered.groupby(['Material', 'Nuevo Canal'])['Venta UMB'].sum().reset_index()
        td_sellin = td_sellin[td_sellin['Venta UMB'] >= 0]
        venta_total = td_sellin['Venta UMB'].sum()

        st.subheader("Resumen Sell In")
        st.metric("Total Venta UMB (positiva)", f"{venta_total:,.0f}")
        st.dataframe(td_sellin)

        two_months_ago = (datetime.now() - pd.DateOffset(months=2)).strftime('%Y-%m')

        uploaded_hist = st.file_uploader("📤 Sube el archivo histórico .parquet", type="parquet")
        if uploaded_hist is None:
            st.warning("⚠️ Esperando archivo histórico...")
            st.stop()
        df_combinado = pd.read_parquet(uploaded_hist)

        from datetime import datetime, timedelta
        # Procesar campo fecha si existe
        if 'Año/Mes natural' in df_combinado.columns:
            df_combinado['Año/Mes natural'] = pd.to_datetime(df_combinado['Año/Mes natural'], format='%d-%m-%Y', errors='coerce')
            df_combinado['Año/Mes natural'] = df_combinado['Año/Mes natural'].dt.strftime('%d-%m-%Y')
        else:
            st.warning("⚠️ La columna 'Año/Mes natural' no existe en el archivo.")

        # Mostrar info general
        st.write("**Shape del DataFrame:**", df_combinado.shape)
        st.write("**Tipos de datos:**")
        st.dataframe(df_combinado.dtypes.reset_index().rename(columns={"index": "Columna", 0: "Tipo"}))

        if 'Venta' in df_combinado.columns:
            suma_venta_umb = df_combinado['Venta'].sum()
            st.metric("Suma columna 'Venta'", f"{suma_venta_umb:,.0f}")

        st.subheader("Vista previa del histórico")
        st.dataframe(df_combinado.head())

        st.subheader("Cargando archivo maestro COD")

        uploaded_cod = st.file_uploader("📤 Sube el archivo COD (.csv)", type="csv")
        if uploaded_cod is None:
         st.warning("⚠️ Esperando archivo COD...")
         st.stop()
        ultimoeslabon = pd.read_csv(uploaded_cod, delimiter=';', encoding='latin-1', decimal=',')

        eslabon = ultimoeslabon[['Nro_pieza_fabricante_1', 'Cod_Actual_1']].copy()
        eslabon.rename(columns={'Nro_pieza_fabricante_1': 'Material'}, inplace=True)
        eslabon['Material'] = eslabon['Material'].astype(str)
        td_sellin['Material'] = td_sellin['Material'].astype(str)

        # Crear nuevas filas
        meses_abreviados = {
            1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Ago', 9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dic'
        }

        nuevas_filas = td_sellin.copy()
        nuevas_filas.rename(columns={'Nuevo Canal': 'Canal', 'Venta UMB': 'Venta'}, inplace=True)

        fecha_actual = datetime.now().replace(day=1)
        fecha_mes_anterior = fecha_actual - timedelta(days=1)
        fecha_mes_anterior = fecha_mes_anterior.replace(day=1)

        fecha_mes_anterior_str = fecha_mes_anterior.strftime('%d-%m-%Y')
        mes_numero = fecha_mes_anterior.month
        mes_abreviado = meses_abreviados[mes_numero]
        ano_actual_yy = fecha_mes_anterior.strftime('%y')

        nuevas_filas['Año/Mes natural'] = fecha_mes_anterior_str
        nuevas_filas['Fuente'] = f"Sell In {mes_abreviado}-{ano_actual_yy}"

        for col in df_combinado.columns:
            if col not in nuevas_filas:
                nuevas_filas[col] = None

        st.subheader("Nuevas filas preparadas")
        st.write(f"Periodo: {fecha_mes_anterior_str} | Fuente: Sell In {mes_abreviado}-{ano_actual_yy}")
        st.dataframe(nuevas_filas.head())

        # Concatenar nuevas filas al histórico
        st.subheader("Integración de nuevas filas al histórico")

        df_combinado_actualizado2 = pd.concat([df_combinado, nuevas_filas], ignore_index=True)

        # Mapeo de Canal 3
        lista_1 = ["Agroplanet", "Autoplanet", "CES 01", "FyC 03", "Mayorista", "May AP", "Sodimac",
                "Walmart", "Easy", "Tottus", "SMU", "Interfilial", "IE", "Colombia", "Bolivia",
                "Recasur", "Dacovi", "GT", "Perú", "Flotas"]
        lista_2 = ["CL AGROPLANET", "CL AUTOPLANET", "CL CES 01", "CL FYC 03", "CL MAYORISTA", "CL MAY AP",
                "CL SODIMAC", "CL WALMART", "CL EASY", "CL TOTTUS", "CL SMU", "NO APLICA", "NO APLICA",
                "NO APLICA", "NO APLICA", "NO APLICA", "NO APLICA", "NO APLICA", "NO APLICA", "CL FYC 03"]
        mapa_canal = dict(zip(lista_1, lista_2))

        indice_nuevas_filas = range(len(df_combinado_actualizado2) - len(nuevas_filas), len(df_combinado_actualizado2))
        df_combinado_actualizado2.loc[indice_nuevas_filas, 'Canal 3'] = df_combinado_actualizado2.loc[indice_nuevas_filas, 'Canal'].map(mapa_canal)

        # Merge con maestro COD
        consolidado = df_combinado_actualizado2.copy()
        consolidado['Ultimo Eslabón'] = None

        merged_df = pd.merge(consolidado, eslabon, on='Material', how='left')
        merged_df['Ultimo Eslabón'] = merged_df['Cod_Actual_1'].fillna(merged_df['Material'])
        merged_df = merged_df[consolidado.columns]
        totalsellin = merged_df

        # Cargar archivo MARA
        st.subheader("Cargando archivo MARA")

        ruta_directorio_mara = os.path.join(
            os.path.expanduser('~'),
            "DERCO CHILE REPUESTOS SpA",
            "Planificación y abastecimiento - Documentos",
            "Planificación y Compras Maestros",
            str(año_actual),
            f"{año_actual}-{mes_actual}",
            "MaestrosCSV"
        )
        archivos_mara = glob.glob(os.path.join(ruta_directorio_mara, "*MARA*.csv"))

        # if archivos_mara:
        #     archivo_mara = archivos_mara[0]
        #     mara = pd.read_csv(archivo_mara, delimiter=';')
        #     st.success(f"✅ Archivo MARA cargado: {os.path.basename(archivo_mara)}")
        # else:
        #     st.error("❌ No se encontró ningún archivo que contenga 'MARA' en el nombre.")
        #     st.stop()
        uploaded_mara = st.file_uploader("📤 Sube el archivo MARA (.csv)", type="csv")
        if uploaded_mara is None:
            st.warning("⚠️ Esperando archivo MARA...")
            st.stop()
        mara = pd.read_csv(uploaded_mara, delimiter=';')

        from datetime import datetime, timedelta
        dfmara = mara.copy()
        dfmara.rename(columns={'Material': 'Material_S4'}, inplace=True)
        dfmara.rename(columns={'Nombre_Sector': 'Nombre Sector'}, inplace=True)
        dfmara['Material_S4'] = dfmara['Material_S4'].astype(str)
        mara_reducido = dfmara[['Material_S4', 'Nombre Sector', 'Sector_MU']]
        totalsellin['Material'] = totalsellin['Material'].apply(lambda x: str(x)[:-2] if '.' in str(x) and str(x).endswith('.0') else x)
        totalsellin['Ultimo Eslabón'] = totalsellin['Ultimo Eslabón'].apply(lambda x: str(x)[:-2] if '.' in str(x) and str(x).endswith('.0') else x)
        totalsellin['Ultimo Eslabón'] = totalsellin['Ultimo Eslabón'].astype(str)
        mara_reducido['Material_S4'] = mara_reducido['Material_S4'].astype(str)
        merged_df4 = pd.merge(totalsellin, mara_reducido, left_on='Ultimo Eslabón', right_on='Material_S4')
        st.success("✅ Unión con sectores completada correctamente")
        df_filtrado = merged_df4[merged_df4['Año/Mes natural'] == '01-11-2024']
        suma_ventas = df_filtrado['Venta'].sum()
        st.subheader("Validación por fecha específica")
        st.metric("Suma de ventas 01-11-2024", f"{suma_ventas:,.0f}")
        merged_df4['Sector'] = merged_df4['Nombre Sector']
        final_df = merged_df4[['Año/Mes natural', 'Material', 'Canal', 'Venta', 'Fuente',
                            'Ultimo Eslabón', 'Reemplazo Manual AFM', 'Nuevo Canal', 'Sector', 'Canal 3']]
        final_df['Sector'] = final_df['Sector'].astype(str)
        df_filtrado = final_df[final_df['Año/Mes natural'] == '01-11-2024']
        suma_ventas = df_filtrado['Venta'].sum()
        st.subheader("Suma de ventas específicas")
        st.metric("Ventas 01-11-2024", f"{suma_ventas:,.0f}")

        st.subheader("Guardando archivo .parquet")
        import io

        hoy = datetime.today()
        primer_dia_mes_actual = hoy.replace(day=1)
        mes_pasado = primer_dia_mes_actual - timedelta(days=1)
        mes_pasado_str = mes_pasado.strftime("%Y-%m")

        # Crear buffer en memoria
        buffer_parquet = io.BytesIO()
        final_df.to_parquet(buffer_parquet, index=False)
        buffer_parquet.seek(0)

        # Botón de descarga
        st.download_button(
            label=f"⬇️ Descargar Historia_Sell_In ({mes_pasado_str}).parquet",
            data=buffer_parquet,
            file_name=f"Historia_Sell_In ({mes_pasado_str}).parquet",
            mime="application/octet-stream"
        )

        # Filtrar por canales y sectores deseados
        st.subheader("Filtrando por canales y sectores")

        canales_deseados = ['CL CES 01', 'CL MAYORISTA']
        sectores_deseados = ['ACC', 'BAT', 'NEU', 'LUB', 'RALT', 'RMAQ']

        final_df = final_df.loc[
            (final_df['Canal 3'].isin(canales_deseados)) &
            (final_df['Sector'].isin(sectores_deseados))
        ]

        st.dataframe(final_df.head())

        # Pivot dinámico por canal y eslabón
        sellinreducido = final_df[['Año/Mes natural', 'Ultimo Eslabón', 'Canal 3', 'Venta']]
        sellinreducido_ = sellinreducido.copy()
        sellinreducido_.rename(columns={'Año/Mes natural': 'Fecha'}, inplace=True)
        sellinreducido_['Venta'] = pd.to_numeric(sellinreducido_['Venta'], errors='coerce')

        pivot_df = sellinreducido_.pivot_table(
            index=['Ultimo Eslabón', 'Canal 3'],
            columns='Fecha',
            values='Venta',
            aggfunc='sum'
        ).reset_index()

        # Limpieza
        pivot_df.fillna(0, inplace=True)
        cols = pivot_df.columns.drop(['Ultimo Eslabón', 'Canal 3'])
        pivot_df[cols] = pivot_df[cols].apply(pd.to_numeric, errors='coerce').fillna(0).clip(lower=0)

        # Eliminar filas con suma total cero
        pivot_df['Suma'] = pivot_df[cols].sum(axis=1)
        pivot_df = pivot_df[pivot_df['Suma'] != 0].drop(columns=['Suma'])

        # Ordenar columnas por fecha
        non_date_cols = ['Ultimo Eslabón', 'Canal 3']
        date_cols = [col for col in pivot_df.columns if col not in non_date_cols]
        date_cols_sorted = sorted(date_cols, key=lambda x: pd.to_datetime(x, format='mixed'))
        pivot_df = pivot_df[non_date_cols + date_cols_sorted]

        # Eliminar columnas con años antiguos
        columns_to_keep = [col for col in pivot_df.columns if not any(y in col for y in ['2015', '2016', '2017'])]
        pivot_df = pivot_df[columns_to_keep]

        st.subheader("Pivot final preparado")
        st.dataframe(pivot_df.head())

        # Export paths
        # st.subheader("Definiendo rutas de exportación")

        # user_dir = os.path.expanduser('~')
        # now = datetime.now()
        # current_year = now.strftime('%Y')
        # current_month = now.strftime('%m')
        # previous_month = (now.replace(day=1) - pd.DateOffset(months=1)).strftime('%B-%y')
        # cycle_month = (now + pd.DateOffset(months=1)).strftime('%b-%y')

        # base_path = os.path.join(
        #     user_dir,
        #     'DERCO CHILE REPUESTOS SpA',
        #     'Planificación y abastecimiento - Documentos',
        #     'Planificación y Compras Anastasia',
        #     'Carga Historia de Venta',
        #     f'{current_year}-{current_month} Ciclo {cycle_month}',
        #     'AFM',
        #     'SELL IN'
        # )
        # os.makedirs(base_path, exist_ok=True)

        # csv_path = os.path.join(base_path, f'{current_month}.{current_year} Sell_In {previous_month} Corregido.csv')
        # excel_path = os.path.join(base_path, f'{current_month}.{current_year} Sell_In {previous_month} Corregido.xlsx')

        # Crear columnas nuevas
        st.subheader("Cálculo de estadísticas avanzadas")

        nuevas_columnas = ['Clasificación', 'PROMEDIO', 'DESV EST', 'Z', 'LIM SUP', 'LIM INF', 'FRECUENCIA',
                        'Outliers SUP', 'Outliers INF', 'Suma Outliers', 'PERCENTIL', 'Clustering', 'ADI',
                        'CV2', 'MEDIANA', 'PROM_CV2', 'Desv_CV2', 'LIM SUP MED', 'LIM INF FINAL',
                        'LIM INF ANT', 'LIM INF MEDIANA']

        new_columns_df = pd.DataFrame(index=pivot_df.index, columns=[col for col in nuevas_columnas if col not in pivot_df.columns])
        pivot_df = pd.concat([pivot_df, new_columns_df], axis=1)

        # Cálculo de métricas
        index_clasificacion = pivot_df.columns.get_loc('Clasificación')
        columnas_conteo = pivot_df.columns[index_clasificacion-18:index_clasificacion]
        pivot_df['PROMEDIO'] = pivot_df[columnas_conteo].mean(axis=1)

        num_cols = 24
        col_pos = index_clasificacion
        cols_for_adi = pivot_df.iloc[:, col_pos - num_cols:col_pos]
        pivot_df['ADI'] = np.where(cols_for_adi.gt(0).sum(axis=1) > 0,
                                24 / cols_for_adi.gt(0).sum(axis=1), 0)

        cols_for_avg = pivot_df.iloc[:, col_pos - num_cols:col_pos]
        pivot_df['PROM_CV2'] = cols_for_avg.replace(0, np.nan).mean(axis=1)

        cols_for_std = pivot_df.iloc[:, col_pos - num_cols:col_pos]
        pivot_df['Desv_CV2'] = cols_for_std.replace(0, np.nan).std(axis=1)

        cols_for_median = pivot_df.iloc[:, col_pos - 18:col_pos]
        pivot_df['MEDIANA'] = cols_for_median.median(axis=1)

        pivot_df['CV2'] = np.where(pivot_df['PROM_CV2'].fillna(0) != 0,
                                (pivot_df['Desv_CV2'].fillna(0) / pivot_df['PROM_CV2'].fillna(0))**2, 0)
        pivot_df['CV2'] = pivot_df['CV2'].fillna(0)

        pivot_df['Clustering'] = np.where((pivot_df['ADI'] < 1.32) & (pivot_df['CV2'] < 0.49), "SMOOTH",
                                np.where((pivot_df['ADI'] >= 1.32) & (pivot_df['CV2'] < 0.49), "INTERMITTENT",
                                np.where((pivot_df['ADI'] < 1.32) & (pivot_df['CV2'] >= 0.49), "ERRATIC",
                                np.where((pivot_df['ADI'] >= 1.32) & (pivot_df['CV2'] >= 0.49), "LUMPY", ""))))

        # Calcular percentil 10 basado en clustering
        st.subheader("Cálculo de percentiles y clasificación")

        import numpy as np

        idx = pivot_df.columns.get_loc("Clasificación")

        pivot_df['PERCENTIL'] = pivot_df.apply(lambda row: 
            np.percentile(row.iloc[idx - 24:idx].values, 10, method='weibull') 
            if row['Clustering'] in ['ERRATIC', 'LUMPY'] 
            else np.percentile(row.iloc[idx - 24:idx].values, 10, method='linear'), 
            axis=1)


        # Calcular FRECUENCIA, PROMEDIO, DESV EST
        columnas_conteo = pivot_df.columns[index_clasificacion-24:index_clasificacion]
        pivot_df['FRECUENCIA'] = (pivot_df[columnas_conteo] > 0).sum(axis=1)

        columnas_conteo = pivot_df.columns[index_clasificacion-18:index_clasificacion]
        pivot_df['PROMEDIO'] = pivot_df[columnas_conteo].mean(axis=1)

        columnas_conteo = pivot_df.columns[index_clasificacion-24:index_clasificacion]
        pivot_df['DESV EST'] = pivot_df[columnas_conteo].std(ddof=1, axis=1)

        # Clasificación por frecuencia
        def clasificar_frecuencia(frecuencia):
            if frecuencia == 0:
                return "Sin Venta"
            elif 0 < frecuencia <= 6:
                return "Muy Baja"
            elif 6 < frecuencia <= 9:
                return "Baja"
            elif 9 < frecuencia <= 13:
                return "Media"
            elif 13 < frecuencia <= 18:
                return "Alta"
            elif 18 < frecuencia <= 24:
                return "Muy Alta"

        pivot_df['Clasificación'] = pivot_df['FRECUENCIA'].apply(clasificar_frecuencia)

        st.success("✅ Cálculos estadísticos completados.")
        st.dataframe(pivot_df[['ADI', 'CV2', 'Clustering', 'FRECUENCIA', 'Clasificación']].head())
        # Cálculo de Z según clasificación
        def calcular_z(clasificacion):
            if clasificacion == "Muy Alta":
                return 1.96
            elif clasificacion == "Alta":
                return 1.28
            elif clasificacion in ["Baja", "Muy Baja"]:
                return 0.67
            else:
                return 0.84

        pivot_df['Z'] = pivot_df['Clasificación'].apply(calcular_z)
        pivot_df['LIM SUP'] = pivot_df['PROMEDIO'] + pivot_df['Z'] * pivot_df['DESV EST']

        # Cálculo del límite inferior con PERCENTIL
        def calcular_lim_inf(row):
            promedio = row['PROMEDIO']
            z = row['Z']
            desv_est = row['DESV EST']
            percentil = row['PERCENTIL']
            resultado = promedio - z * desv_est
            return max(resultado, percentil)

        pivot_df['LIM INF'] = pivot_df.apply(calcular_lim_inf, axis=1)

        # Cálculo de outliers
        index_clasificacion = pivot_df.columns.get_loc('Clasificación')
        columnas_conteo = pivot_df.columns[index_clasificacion-18:index_clasificacion]

        pivot_df['Outliers SUP'] = pivot_df.apply(
            lambda row: (row[columnas_conteo] > row['LIM SUP']).sum(), axis=1
        )

        pivot_df['Outliers INF'] = pivot_df.apply(
            lambda row: (row[columnas_conteo] < row['LIM INF']).sum(), axis=1
        )

        pivot_df['Suma Outliers'] = pivot_df['Outliers INF'] + pivot_df['Outliers SUP']

        # Copiar columnas previas a Clasificación
        clasificacion_index = pivot_df.columns.get_loc("Clasificación")
        columns_to_copy = [col for col in pivot_df.columns[:clasificacion_index] if col not in ["idSKU", "Canal"]]
        copied_columns = pivot_df[columns_to_copy].copy()
        copied_columns.columns = [f'{col}_copy' for col in copied_columns.columns]
        pivot_df = pd.concat([pivot_df, copied_columns], axis=1)

        # Aplicar limpieza a últimas 18 columnas
        last_18_columns = pivot_df.columns[-18:]
        for col in last_18_columns:
            pivot_df[col] = pivot_df.apply(
                lambda row: row['PROMEDIO'] if row[col] > row['LIM SUP'] or row[col] < row['LIM INF'] else row[col],
                axis=1
            )

        # Sumar última columna
        last_column = pivot_df.columns[-1]
        sum_last_column = pivot_df[last_column].sum()
        st.metric(f"Suma última columna ({last_column})", f"{sum_last_column:,.0f}")

        # Eliminar columnas auxiliares post-outliers
        index_suma_outliers = pivot_df.columns.get_loc("Suma Outliers")
        cols_to_drop = [col for col in pivot_df.columns[index_suma_outliers+1:] if col.startswith("Ul") or col.startswith("Ca")]
        pivot_df.drop(columns=cols_to_drop, inplace=True)

        # Renombrar columnas "_copy"
        new_columns = pivot_df.columns.tolist()
        for i in range(index_suma_outliers + 1, len(new_columns)):
            if new_columns[i].endswith("_copy"):
                new_columns[i] = new_columns[i].replace("_copy", "")
        pivot_df.columns = new_columns

        # # Exportar CSV y Excel
        # pivot_df.to_csv(csv_path, sep=';', decimal=',', index=False)
        # pivot_df.to_excel(excel_path, index=False)

        # st.success("✅ Archivos exportados correctamente")
        # st.caption(f"📁 Archivo CSV guardado en: `{csv_path}`")
        # st.caption(f"📁 Archivo Excel guardado en: `{excel_path}`")
        import io

        # Exportar a CSV (en memoria)
        buffer_csv = io.StringIO()
        pivot_df.to_csv(buffer_csv, sep=';', decimal=',', index=False)
        csv_bytes = io.BytesIO(buffer_csv.getvalue().encode('utf-8'))

        # Exportar a Excel (en memoria)
        buffer_excel = io.BytesIO()
        with pd.ExcelWriter(buffer_excel, engine='xlsxwriter') as writer:
            pivot_df.to_excel(writer, index=False)
        buffer_excel.seek(0)

        # Mostrar botones de descarga
        st.success("✅ Archivos exportados correctamente")

        st.download_button(
            label="⬇️ Descargar CSV Corregido",
            data=csv_bytes,
            file_name="Sell_In_Corregido.csv",
            mime="text/csv"
        )

        st.download_button(
            label="⬇️ Descargar Excel Corregido",
            data=buffer_excel,
            file_name="Sell_In_Corregido.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )



def corregir_sellout():
        import pandas as pd
        import os
        import pandas as pd
        import numpy as np
        import glob
        from datetime import datetime, timedelta

        st.subheader("Cargando histórico Sell Out")
        # two_months_ago = (datetime.now() - pd.DateOffset(months=2)).strftime('%Y-%m')
        # ruta_automatica = os.path.join(
        #     os.path.expanduser('~'),
        #     'DERCO CHILE REPUESTOS SpA',
        #     'Planificación y abastecimiento - Documentos',
        #     'Planificación y Compras AFM',
        #     'S&OP Demanda',
        #     'Codigos Demanda',
        #     'Parquets',
        #     f'Historia_Sell_Out ({two_months_ago})_Canales.parquet'
        # )
        # sellout_concat = pd.read_parquet(ruta_automatica)
        # st.success(f"✅ Histórico cargado desde: {ruta_automatica}")
        archivo_hist = st.file_uploader("📤 Subir archivo histórico Sell Out (.parquet)", type="parquet")

        if archivo_hist is not None:
            sellout_concat = pd.read_parquet(archivo_hist)
            st.success("✅ Histórico Sell Out cargado correctamente")
            st.dataframe(sellout_concat.head())
        else:
            st.stop()


        # st.subheader("Cargando archivo Sell Out GT")
        # fecha_actual = pd.Timestamp.now()
        # año_actual = fecha_actual.year
        # mes_actual = fecha_actual.strftime("%m")
        # ruta_directorio = os.path.join(
        #     os.path.expanduser("~"),
        #     "DERCO CHILE REPUESTOS SpA",
        #     "Planificación y abastecimiento - Documentos",
        #     "Planificación y Compras Ventas",
        #     "Venta Historica Mensual",
        #     str(año_actual),
        #     f"{año_actual}-{mes_actual}"
        # )

        # archivo_selloutGT = next((f for f in os.listdir(ruta_directorio) if "Sell Out" in f and "GT" in f), None)

        # if archivo_selloutGT:
        #     ruta_archivo_selloutGT = os.path.join(ruta_directorio, archivo_selloutGT)
        #     selloutGT = pd.read_excel(ruta_archivo_selloutGT, sheet_name="Sell Out GT", header=2)
        #     columnas_a_mantener = [selloutGT.columns[-2], selloutGT.columns[-1], selloutGT.columns[0], selloutGT.columns[1]]
        #     selloutGT1 = selloutGT[columnas_a_mantener]
        #     selloutGT1.rename(columns={selloutGT1.columns[0]: 'Venta'}, inplace=True)
        #     selloutGT1['Venta'] = selloutGT1['Venta'].astype(int)
        #     selloutGT1 = selloutGT1[selloutGT1['Venta'] > 0]
        #     selloutGT1 = selloutGT1[~selloutGT1['Material S4'].str.contains('Total', case=False, na=False)]
        #     selloutGT1['Fuente'] = 'Sell Out MB51 - SISO'
        #     selloutGT1['SI/SO'] = 'SO'
        #     primer_dia_mes_anterior = (fecha_actual.replace(day=1) - timedelta(days=1)).replace(day=1)
        #     selloutGT1['Mes'] = primer_dia_mes_anterior.strftime('%Y-%m-%d')

        #     v1 = selloutGT1['Venta'].sum()
        #     st.metric("Suma Sell Out GT", f"{v1:,.0f}")
        #     st.success(f"✅ Archivo cargado: {ruta_archivo_selloutGT}")
        # else:
        #     st.error("❌ No se encontró archivo de Sell Out GT")
        #     st.stop()


        # st.subheader("Unificando Sell Out GT al histórico")
        # selloutGT1.rename(columns={'Material S4': 'Material', 'Venta': 'Venta UMB', 'Cliente': 'Canal'}, inplace=True)
        # selloutGT1 = selloutGT1.drop_duplicates(subset=['Material', 'Canal', 'Mes'])
        # columnas_comunes = [col for col in selloutGT1.columns if col in sellout_concat.columns]
        # selloutGT1 = selloutGT1[columnas_comunes]

        # st.metric("Total histórico actualizado", f"{sellout_concat['Venta UMB'].sum():,.0f}")
        # st.dataframe(sellout_concat.head())

        # st.subheader("Cargando archivo Sell Out General")
        # archivo_sellout = next((f for f in os.listdir(ruta_directorio) if "Sell Out" in f and "GT" not in f), None)
        # if archivo_sellout:
        #     ruta_archivo_sellout = os.path.join(ruta_directorio, archivo_sellout)
        #     sellout = pd.read_excel(ruta_archivo_sellout, sheet_name="Sheet1")
        #     st.success(f"✅ Sell Out general leído: {archivo_sellout}")
        #     columnas_requeridas = ['Ce.', 'Material', 'Texto breve de material', 'Fe.contab.', 'Cliente',
        #                         'Ult.Eslabón', 'Nombre Sector', 'Canal', 'Tipo', 'Forecast AFM',
        #                         'Venta UMB', 'Semana', 'Canal 2', 'ID']
        #     sellin = sellout[columnas_requeridas]
        #     st.dataframe(sellin.head(1))
        # else:
        #     st.error("❌ No se encontró archivo de Sell Out general")
        
        #     st.subheader("Procesando Sell Out General")
        st.subheader("📤 Cargar archivo Sell Out GT (.xlsx)")
        archivo_selloutGT = st.file_uploader("Sube el archivo Sell Out GT", type="xlsx", key="sellout_gt")

        if archivo_selloutGT is not None:
            fecha_actual = pd.Timestamp.now()
            selloutGT = pd.read_excel(archivo_selloutGT, sheet_name="Sell Out GT", header=2)
            columnas_a_mantener = [selloutGT.columns[-2], selloutGT.columns[-1], selloutGT.columns[0], selloutGT.columns[1]]
            selloutGT1 = selloutGT[columnas_a_mantener]
            selloutGT1.rename(columns={selloutGT1.columns[0]: 'Venta'}, inplace=True)
            selloutGT1['Venta'] = selloutGT1['Venta'].astype(int)
            selloutGT1 = selloutGT1[selloutGT1['Venta'] > 0]
            selloutGT1 = selloutGT1[~selloutGT1['Material S4'].str.contains('Total', case=False, na=False)]
            selloutGT1['Fuente'] = 'Sell Out MB51 - SISO'
            selloutGT1['SI/SO'] = 'SO'
            primer_dia_mes_anterior = (fecha_actual.replace(day=1) - timedelta(days=1)).replace(day=1)
            selloutGT1['Mes'] = primer_dia_mes_anterior.strftime('%Y-%m-%d')

            v1 = selloutGT1['Venta'].sum()
            st.metric("Suma Sell Out GT", f"{v1:,.0f}")
            st.success("✅ Archivo Sell Out GT cargado correctamente")
        else:
            st.stop()

        st.subheader("📤 Cargar archivo Sell Out General (Ultimo mes de venta cerrado) (.xlsx)")
        archivo_sellout = st.file_uploader("Sube el archivo Sell Out General", type="xlsx", key="sellout_general")

        if archivo_sellout is not None:
            sellout = pd.read_excel(archivo_sellout, sheet_name="Sheet1")
            st.success("✅ Sell Out general leído correctamente")
            columnas_requeridas = ['Ce.', 'Material', 'Texto breve de material', 'Fe.contab.', 'Cliente',
                                'Ult.Eslabón', 'Nombre Sector', 'Canal', 'Tipo', 'Forecast AFM',
                                'Venta UMB', 'Semana', 'Canal 2', 'ID']
            sellin = sellout[columnas_requeridas]
            st.dataframe(sellin.head(1))
        else:
            st.error("❌ Debes subir el archivo Sell Out del último mes")
            st.stop()


        df_sellin = sellin.copy()
        df_sellin.rename(columns={'Ult.EslabÃ³n': 'Ult.Eslabón'}, inplace=True)
        df_sellin['Material'] = df_sellin['Material'].astype(str).str.replace('.0', '', regex=False)

        td_sellin = df_sellin.groupby(['Material', 'Canal'])['Venta UMB'].sum().reset_index()
        td_sellin['Venta UMB'] = pd.to_numeric(td_sellin['Venta UMB'], errors='coerce')
        td_sellin = td_sellin[td_sellin['Venta UMB'] > 0]

        st.metric("Venta UMB total filtrada", f"{td_sellin['Venta UMB'].sum():,.0f}")
        st.write("Canales únicos en Sell Out general:")
        st.write(td_sellin['Canal'].unique())

        st.subheader("Cargando y mapeando histórico Sell Out")
        df_combinado = sellout_concat.copy()
        canal_mapping = {
            'AP': 'Autoplanet', 'Sergo': 'Agroplanet', 'Easy': 'Easy', 'TOTTUS': 'TOTTUS',
            'Walmart': 'Walmart', 'Sodimac': 'Sodimac', 'SMU': 'SMU',
            'Autoplanet': 'Autoplanet', 'Agroplanet': 'Agroplanet'
        }
        df_combinado['Canal 2'] = df_combinado['Canal'].map(canal_mapping)
        st.write("Valores únicos mapeados:", df_combinado['Canal 2'].unique())

        # st.subheader("Cargando archivo COD actual")
        # user_dir = os.path.expanduser("~")
        # now = datetime.now()
        # year_month = now.strftime('%Y-%m')
        # base_path = os.path.join(user_dir, 'DERCO CHILE REPUESTOS SpA',
        #                         'Planificación y abastecimiento - Documentos',
        #                         'Planificación y Compras Maestros',
        #                         str(now.year), year_month, 'MaestrosCSV')
        # pattern = os.path.join(base_path, '*COD_ACTUAL*.csv')
        # files = [f for f in glob.glob(pattern) if 'R3' not in os.path.basename(f)]

        # if not files:
        #     st.warning("⚠️ No se encontró archivo COD_ACTUAL")
        #     return
        archivo_cod = st.file_uploader("📤 Subir archivo COD ACTUAL (.csv)", type="csv")

        if archivo_cod is not None:
            eslabon = pd.read_csv(archivo_cod, delimiter=';', decimal=',', dtype=str)[['Nro_pieza_fabricante_1', 'Cod_Actual_1']]
            eslabon.rename(columns={'Nro_pieza_fabricante_1': 'Material'}, inplace=True)
            ...
        else:
            st.error("❌ Debes subir el archivo COD ACTUAL")
            st.stop()


        # latest_file = max(files, key=os.path.getmtime)
        # eslabon = pd.read_csv(latest_file, delimiter=';', decimal=',', low_memory=False, dtype=str)[['Nro_pieza_fabricante_1', 'Cod_Actual_1']]
        # eslabon.rename(columns={'Nro_pieza_fabricante_1': 'Material'}, inplace=True)

        # st.success(f"✅ COD cargado: {os.path.basename(latest_file)}")

        td_sellin['Material'] = td_sellin['Material'].astype(str).str.replace('.0', '', regex=False)
        td_sellin = td_sellin.groupby(['Material', 'Canal'])[['Venta UMB']].sum().reset_index()

        st.subheader("Generando nuevas filas Sell Out")
        meses_abreviados = {
            1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Ago', 9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dic'
        }

        nuevas_filas = td_sellin.copy()
        fecha_actual = datetime.now().replace(day=1)
        fecha_mes_anterior = (fecha_actual - timedelta(days=1)).replace(day=1)
        fecha_mes_anterior_str = fecha_mes_anterior.strftime('%d-%m-%Y')
        mes_abreviado = meses_abreviados[fecha_mes_anterior.month]
        ano_actual_yy = fecha_mes_anterior.strftime('%y')

        nuevas_filas['Fuente'] = f"Sell Out RTL {mes_abreviado}-{ano_actual_yy}"
        nuevas_filas['SI/SO'] = 'SO'
        nuevas_filas['Mes'] = fecha_mes_anterior.strftime('%Y-%m-%d')
        nuevas_filas['Ultimo Eslabón'] = None
        nuevas_filas['Tipo'] = None
        nuevas_filas['Canal 2'] = nuevas_filas['Canal'].apply(lambda x: 'Agroplanet' if x in ['Agroplanet', 'Autoplanet'] else 'Revisar')
        nuevas_filas['Canal 3'] = None

        # Ajustar columnas para que coincidan
        columnas_finales = ['Canal', 'Material', 'Venta UMB', 'Mes', 'Ultimo Eslabón',
                            'Tipo', 'Fuente', 'Canal 2', 'Canal 3', 'SI/SO']
        nuevas_filas = nuevas_filas[columnas_finales]

        st.metric("Total nueva venta RTL", f"{nuevas_filas['Venta UMB'].sum():,.0f}")

        st.subheader("Concatenando histórico + nuevas filas")
        df_combinado_actualizado2 = pd.concat([df_combinado, nuevas_filas], ignore_index=True)
        df_combinado_actualizado2['Material'] = df_combinado_actualizado2['Material'].astype(str).str.replace('.0', '', regex=False)
        df_combinado_actualizado2['Mes'] = pd.to_datetime(df_combinado_actualizado2['Mes'], errors='coerce')

        abril_count = df_combinado_actualizado2[df_combinado_actualizado2['Mes'] == "2024-04-01"].shape[0]
        abril_fuente_count = df_combinado_actualizado2[df_combinado_actualizado2['Fuente'].str.contains("Abr-24", na=False)].shape[0]

        st.metric("Filas con Mes = 2024-04-01", abril_count)
        st.metric("Filas con Fuente Abr-24", abril_fuente_count)
        st.metric("Venta total combinada", f"{df_combinado_actualizado2['Venta UMB'].sum():,.0f}")

        consolidado = df_combinado_actualizado2
        st.success("✅ Consolidado Sell Out generado correctamente")
        st.dataframe(consolidado.tail(2))
        st.subheader("Consolidando y enriqueciendo información final")

        consolidado['Ultimo Eslabón'] = None
        merged_df = pd.merge(consolidado, eslabon, on='Material', how='left')
        merged_df['Ultimo Eslabón'] = merged_df['Cod_Actual_1'].fillna(merged_df['Material'])
        merged_df = merged_df[consolidado.columns]
        totalsellin = merged_df.copy()

        selloutGT1['Mes'] = pd.to_datetime(selloutGT1['Mes'])
        totalsellin = pd.concat([totalsellin, selloutGT1], ignore_index=True)

        st.metric("Suma total Venta UMB", f"{totalsellin['Venta UMB'].sum():,.0f}")
        st.caption("✅ Consolidado unido con Sell Out GT")

        # usuario_dir = os.path.expanduser("~")
        # año_actual = datetime.now().year
        # mes_actual = datetime.now().strftime('%m')
        # ruta = os.path.join(usuario_dir, "DERCO CHILE REPUESTOS SpA", "Planificación y abastecimiento - Documentos", "Planificación y Compras Maestros", str(año_actual), f"{año_actual}-{mes_actual}", "MaestrosCSV")

        # archivo_mara = None
        # fecha_modificacion = None
        # for archivo in os.listdir(ruta):
        #     if 'MARA' in archivo and archivo.endswith('.csv'):
        #         ruta_completa = os.path.join(ruta, archivo)
        #         modificacion_actual = os.path.getmtime(ruta_completa)
        #         if fecha_modificacion is None or modificacion_actual > fecha_modificacion:
        #             archivo_mara = ruta_completa
        #             fecha_modificacion = modificacion_actual

        # if archivo_mara:
        #     dfmara = pd.read_csv(archivo_mara, delimiter=';', decimal=',', low_memory=False, dtype=str)
        #     st.success("✅ Archivo MARA cargado")
        # else:
        #     st.warning("⚠️ No se encontró archivo MARA")
        archivo_mara = st.file_uploader("📤 Subir archivo MARA (.csv)", type="csv")

        if archivo_mara is not None:
            dfmara = pd.read_csv(archivo_mara, delimiter=';', decimal=',', dtype=str)
            ...
        else:
            st.warning("⚠️ Debes subir el archivo MARA")
            st.stop()

        mara_reducido = dfmara.rename(columns={'Material': 'Material_S4'})[['Material_S4', 'Nombre Sector', 'Sector_MU']]
        mara_reducido['Material_S4'] = mara_reducido['Material_S4'].astype(str)
        totalsellin = totalsellin.drop(columns=['Ultimo Eslabón'], errors='ignore')
        totalsellin = totalsellin.merge(eslabon, on='Material', how='left')
        totalsellin['Cod_Actual_1'] = totalsellin['Cod_Actual_1'].fillna(totalsellin['Material'])
        totalsellin = totalsellin.rename(columns={'Cod_Actual_1': 'Ultimo Eslabón'})
        totalsellin.drop('Nombre Sector', axis=1, inplace=True)

        merged_df4 = pd.merge(totalsellin, mara_reducido, left_on='Ultimo Eslabón', right_on='Material_S4')
        st.write("🔍 Columnas de merged_df4:", mara_reducido.columns.tolist())  # DEBUG ANTES DE FINAL_DF
        st.write("🔍 Columnas de merged_df4:", totalsellin.columns.tolist())  # DEBUG ANTES DE FINAL_DF
        st.write("🔍 Columnas de merged_df4:", merged_df4.columns.tolist())  # DEBUG ANTES DE FINAL_DF

        merged_df4 = merged_df4.rename(columns={'Nombre Sector': 'Nombre Sector'})  # <- esta línea puede omitirse

        final_df = merged_df4[['Mes', 'Material', 'Canal', 'Venta UMB', 'Fuente', 'Ultimo Eslabón', 'Tipo', 'Canal 2', 'Nombre Sector', 'Canal 3']].copy()
        final_df['Canal 3'] = final_df['Canal'].map({
            'AP': 'CL AUTOPLANET', 'Sergo': 'CL AGROPLANET', 'Easy': 'CL EASY', 'SMU': 'CL SMU',
            'TOTTUS': 'CL TOTTUS', 'Walmart': 'CL WALMART', 'Sodimac': 'CL SODIMAC',
            'Autoplanet': 'CL AUTOPLANET', 'Agroplanet': 'CL AGROPLANET'
        })


        na_count = final_df['Canal 3'].isna().sum()
        st.metric("Canales sin Canal 3", na_count)

        st.dataframe(final_df.head(2))
        mes_pasado = (datetime.today().replace(day=1) - timedelta(days=1)).strftime('%Y-%m')
        import io

        st.dataframe(final_df.head(2))
        mes_pasado = (datetime.today().replace(day=1) - timedelta(days=1)).strftime('%Y-%m')
        nombre_archivo = f"Historia_Sell_Out ({mes_pasado})_Canales.parquet"

        # Guardar en buffer de memoria
        buffer_parquet = io.BytesIO()
        final_df.to_parquet(buffer_parquet, index=False)
        buffer_parquet.seek(0)

        st.download_button(
            label="⬇️ Descargar archivo .parquet",
            data=buffer_parquet,
            file_name=nombre_archivo,
            mime="application/octet-stream"
        )

        # Filtro por sectores deseados
        sectores_deseados = ['ACC', 'BAT', 'NEU', 'LUB', 'RALT', 'RMAQ']
        final_df = final_df[final_df['Nombre Sector'].isin(sectores_deseados)]
        st.dataframe(final_df.head(2))
        import pandas as pd
        from datetime import datetime

        fecha_objetivo = '2024-08-01'
        suma_venta_umb = final_df[final_df['Mes'] == fecha_objetivo]['Venta UMB'].sum()
        print(suma_venta_umb)

        final_df = final_df.drop(columns=['Ultimo Eslabón'], errors='ignore')
        final_df = final_df.merge(eslabon, on='Material', how='left')
        final_df['Cod_Actual_1'] = final_df['Cod_Actual_1'].fillna(final_df['Material'])
        final_df = final_df.rename(columns={'Cod_Actual_1': 'Ultimo Eslabón'})

        sellinreducido = final_df[['Mes', 'Ultimo Eslabón', 'Canal 3', 'Venta UMB']]

        suma_venta_umb = sellinreducido[sellinreducido['Mes'] == fecha_objetivo]['Venta UMB'].sum()
        print(suma_venta_umb)

        sellinreducido_ = sellinreducido.copy()
        sellinreducido_ = sellinreducido_.rename(columns={'Mes': 'Fecha', 'Ultimo Eslabón': 'Material'})
        sellinreducido_['Venta UMB'] = pd.to_numeric(sellinreducido_['Venta UMB'], errors='coerce')

        merged_df1 = sellinreducido_.copy()

        fecha_objetivo = '2024-11-01'
        suma_venta_umb = merged_df1[merged_df1['Fecha'] == fecha_objetivo]['Venta UMB'].sum()
        print(suma_venta_umb)

        sellinreducido1 = merged_df1.rename(columns={'Material': 'Ultimo Eslabón'})

        pivot_df = sellinreducido1.pivot_table(index=['Ultimo Eslabón', 'Canal 3'], columns='Fecha', values='Venta UMB', aggfunc='sum').reset_index()
        pivot_df.fillna(0, inplace=True)

        cols = pivot_df.columns.drop(['Ultimo Eslabón', 'Canal 3'])
        pivot_df[cols] = pivot_df[cols].apply(pd.to_numeric, errors='coerce').fillna(0).clip(lower=0)

        non_date_cols = ['Ultimo Eslabón', 'Canal 3']
        date_cols = [col for col in pivot_df.columns if col not in non_date_cols]
        date_cols_sorted = sorted(date_cols, key=lambda x: pd.to_datetime(x, format='mixed'))

        new_order = non_date_cols + date_cols_sorted
        pivot_df = pivot_df[new_order]

        columns_to_keep = [col for col in pivot_df.columns if not any(year in str(col) for year in ['2015', '2016', '2017'])]
        pivot_df = pivot_df[columns_to_keep]

        # user_dir = os.path.expanduser('~')
        # now = datetime.now()
        # current_year = now.strftime('%Y')
        # current_month = now.strftime('%m')
        # previous_month = (now.replace(day=1) - pd.DateOffset(months=1)).strftime('%B-%y')
        # cycle_month = (now + pd.DateOffset(months=1)).strftime('%b-%y')

        # base_path = os.path.join(
        #     user_dir,
        #     'DERCO CHILE REPUESTOS SpA',
        #     'Planificación y abastecimiento - Documentos',
        #     'Planificación y Compras Anastasia',
        #     'Carga Historia de Venta',
        #     f'{current_year}-{current_month} Ciclo {cycle_month}',
        #     'AFM',
        #     'SELL OUT'
        # )
        # os.makedirs(base_path, exist_ok=True)
        # st.subheader("Aplicación de reglas y guardado de resultados")

        # # Exportar paths
        # csv_path = os.path.join(base_path, f"{current_month}.{current_year} Sell_Out {previous_month} Corregido.csv")
        # excel_path = os.path.join(base_path, f"{current_month}.{current_year} Sell_Out {previous_month} Corregido.xlsx")
        # st.caption(f"📁 Archivos serán guardados en: {base_path}")

        pivot_df['Suma'] = pivot_df.drop(columns=['Ultimo Eslabón', 'Canal 3']).sum(axis=1)
        pivot_df = pivot_df[pivot_df['Suma'] != 0].drop(columns=['Suma'])


        nuevas_columnas = ['Clasificación', 'PROMEDIO', 'DESV EST', 'Z', 'LIM SUP', 'LIM INF', 'FRECUENCIA', 
                        'Outliers SUP', 'Outliers INF', 'Suma Outliers', 'PERCENTIL', 'Clustering', 'ADI', 
                        'CV2', 'MEDIANA', 'PROM_CV2', 'Desv_CV2', 'LIM SUP MED', 'LIM INF FINAL', 
                        'LIM INF ANT', 'LIM INF MEDIANA']
        pivot_df = pd.concat([pivot_df, pd.DataFrame(index=pivot_df.index, columns=[col for col in nuevas_columnas if col not in pivot_df.columns])], axis=1)

        idx = pivot_df.columns.get_loc('Clasificación')
        cols_hist = pivot_df.columns[idx - 24:idx]
        cols_histz = pivot_df.columns[idx - 18:idx]
        pivot_df['PROMEDIO'] = pivot_df[cols_histz].mean(axis=1)
        pivot_df['ADI'] = np.where(pivot_df[cols_hist].gt(0).sum(axis=1) > 0, 24 / pivot_df[cols_hist].gt(0).sum(axis=1), 0)
        pivot_df['PROM_CV2'] = pivot_df[cols_hist].replace(0, np.nan).mean(axis=1)
        pivot_df['Desv_CV2'] = pivot_df[cols_hist].replace(0, np.nan).std(axis=1)
        pivot_df['MEDIANA'] = pivot_df[cols_histz].median(axis=1)
        pivot_df['CV2'] = np.where(pivot_df['PROM_CV2'].fillna(0) != 0, (pivot_df['Desv_CV2'].fillna(0) / pivot_df['PROM_CV2'].fillna(0))**2, 0)

        pivot_df['Clustering'] = np.select(
            [ (pivot_df['ADI'] < 1.32) & (pivot_df['CV2'] < 0.49),
            (pivot_df['ADI'] >= 1.32) & (pivot_df['CV2'] < 0.49),
            (pivot_df['ADI'] < 1.32) & (pivot_df['CV2'] >= 0.49),
            (pivot_df['ADI'] >= 1.32) & (pivot_df['CV2'] >= 0.49) ],
            ['SMOOTH', 'INTERMITTENT', 'ERRATIC', 'LUMPY'], default='')

        pivot_df['PERCENTIL'] = pivot_df.apply(lambda row: np.percentile(row[cols_hist].values, 10, method='weibull') if row['Clustering'] in ['ERRATIC', 'LUMPY'] else np.percentile(row[cols_hist].values, 10, method='linear'), axis=1)
        pivot_df['FRECUENCIA'] = (pivot_df[cols_hist] > 0).sum(axis=1)
        pivot_df['DESV EST'] = pivot_df[cols_hist].std(ddof=1, axis=1)

        pivot_df['Clasificación'] = pivot_df['FRECUENCIA'].apply(lambda x: "Sin Venta" if x == 0 else "Muy Baja" if x <= 6 else "Baja" if x <= 9 else "Media" if x <= 13 else "Alta" if x <= 18 else "Muy Alta")
        pivot_df['Z'] = pivot_df['Clasificación'].map({"Muy Alta": 1.96, "Alta": 1.28, "Baja": 0.67, "Muy Baja": 0.67}).fillna(0.84)
        pivot_df['LIM SUP'] = pivot_df['PROMEDIO'] + pivot_df['Z'] * pivot_df['DESV EST']
        pivot_df['LIM INF'] = pivot_df.apply(lambda row: max(row['PROMEDIO'] - row['Z'] * row['DESV EST'], row['PERCENTIL']), axis=1)

        pivot_df['Outliers SUP'] = pivot_df.apply(lambda row: (row[cols_hist] > row['LIM SUP']).sum(), axis=1)
        pivot_df['Outliers INF'] = pivot_df.apply(lambda row: (row[cols_hist] < row['LIM INF']).sum(), axis=1)
        pivot_df['Suma Outliers'] = pivot_df['Outliers INF'] + pivot_df['Outliers SUP']

        idx = pivot_df.columns.get_loc("Clasificación")
        cols_copiar = [col for col in pivot_df.columns[:idx] if col not in ["idSKU", "Canal"]]
        pivot_df = pd.concat([pivot_df, pivot_df[cols_copiar].copy().rename(columns=lambda x: f"{x}_copy")], axis=1)
        ult_18 = pivot_df.columns[-18:]
        for col in ult_18:
            pivot_df[col] = pivot_df.apply(lambda row: row['PROMEDIO'] if row[col] > row['LIM SUP'] or row[col] < row['LIM INF'] else row[col], axis=1)
        last_col = pivot_df.columns[-1]
        print(f"La suma de la última columna ({last_col}) es: {pivot_df[last_col].sum():,.0f}")

        idx_out = pivot_df.columns.get_loc("Suma Outliers")
        cols_del = [col for col in pivot_df.columns[idx_out+1:] if col.startswith("Ul") or col.startswith("Ca")]
        pivot_df.drop(columns=cols_del, inplace=True)

        nuevas_columnas = pivot_df.columns.tolist()
        for i in range(idx_out + 1, len(nuevas_columnas)):
            if nuevas_columnas[i].endswith("_copy"):
                nuevas_columnas[i] = nuevas_columnas[i].replace("_copy", "")
        pivot_df.columns = nuevas_columnas

        # Guardado final
        # pivot_df.to_csv(csv_path, sep=';', decimal=',', index=False)
        # pivot_df.to_excel(excel_path, index=False)
        # st.success("✅ Corrección de Sell Out completada")}
        import io

        now = datetime.now()
        current_year = now.strftime('%Y')
        current_month = now.strftime('%m')
        previous_month = (now.replace(day=1) - pd.DateOffset(months=1)).strftime('%B-%y')
        cycle_month = (now + pd.DateOffset(months=1)).strftime('%b-%y')

        st.subheader("Aplicación de reglas y guardado de resultados")
        csv_buffer = io.StringIO()
        pivot_df.to_csv(csv_buffer, sep=';', decimal=',', index=False)
        csv_bytes = csv_buffer.getvalue().encode('utf-8')

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            pivot_df.to_excel(writer, index=False)
        excel_buffer.seek(0)
        csv_filename = f"{current_month}.{current_year} Sell_Out {previous_month} Corregido.csv"
        excel_filename = f"{current_month}.{current_year} Sell_Out {previous_month} Corregido.xlsx"

        st.success("✅ Corrección de Sell Out completada")

        st.download_button(
            label=f"⬇️ Descargar CSV: {csv_filename}",
            data=csv_bytes,
            file_name=csv_filename,
            mime="text/csv"
        )

        st.download_button(
            label=f"⬇️ Descargar Excel: {excel_filename}",
            data=excel_buffer,
            file_name=excel_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )



def corregir_disponibilidad():
        import streamlit as st
        import os
        import pandas as pd
        from datetime import datetime
        import glob
        import numpy as np
        import locale

        locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')
        usuario = os.path.expanduser('~')
        fecha_actual = datetime.now().strftime('%Y-%m')

        # ruta_base = os.path.join(
        #     usuario,
        #     'derco chile repuestos spa',
        #     'Planificación y abastecimiento - Documentos',
        #     'Planificación y Compras Anastasia',
        #     'Carga Historia de Venta'
        # )

        # if not os.path.exists(ruta_base):
        #     st.error(f"❌ Ruta base no existe: {ruta_base}")
        #     return

        # carpeta_año_mes = glob.glob(os.path.join(ruta_base, f'{fecha_actual}*'))
        # if not carpeta_año_mes:
        #     st.warning(f"📂 No se encontró carpeta año-mes con prefijo: {fecha_actual}")
        #     return

        # ruta_afm = os.path.join(carpeta_año_mes[0], 'AFM')
        # if not os.path.exists(ruta_afm):
        #     st.error(f"❌ Ruta AFM no encontrada: {ruta_afm}")
        #     return

        # ruta_sell_in = os.path.join(ruta_afm, 'SELL IN')
        # archivos_sell_in = glob.glob(os.path.join(ruta_sell_in, '*Sell_In*.csv'))
        # if not archivos_sell_in:
        #     st.warning(f"❌ No se encontró archivo Sell_In en: {ruta_sell_in}")
        #     return
        # archivo_sell_in = archivos_sell_in[0]
        # st.success(f"📄 Archivo Sell_In cargado: {archivo_sell_in}")
        # df_sell_in = pd.read_csv(archivo_sell_in, delimiter=';', decimal=',')
        archivo_sell_in = st.file_uploader("📤 Carga archivo Sell In (venta cerrada mes pasado)(.csv)", type=["csv"])
        if archivo_sell_in is None:
            st.warning("❗ Esperando archivo Sell In")
            st.stop()
        df_sell_in = pd.read_csv(archivo_sell_in, delimiter=';', decimal=',')


        # ruta_sell_out = os.path.join(ruta_afm, 'SELL OUT')
        # archivos_sell_out = glob.glob(os.path.join(ruta_sell_out, '*Sell_Out*.csv'))
        # if not archivos_sell_out:
        #     st.warning(f"❌ No se encontró archivo Sell_Out en: {ruta_sell_out}")
        #     return
        # archivo_sell_out = archivos_sell_out[0]
        # st.success(f"📄 Archivo Sell_Out cargado: {archivo_sell_out}")
        # df_sell_out = pd.read_csv(archivo_sell_out, delimiter=';', decimal=',')
        archivo_sell_out = st.file_uploader("📤 Carga archivo Sell Out (.csv)", type=["csv"])
        if archivo_sell_out is None:
            st.warning("⚠️ Esperando archivo Sell Out...")
            st.stop()
        df_sell_out = pd.read_csv(archivo_sell_out, delimiter=';', decimal=',')
        st.success("✅ Archivo Sell Out cargado correctamente")


        columnas_a_eliminar = [
            'PERCENTIL', 'Clustering', 'ADI', 'CV2', 'MEDIANA', 'PROM_CV2',
            'Desv_CV2', 'LIM SUP MED', 'LIM INF FINAL', 'LIM INF ANT', 'LIM INF MEDIANA'
        ]
        df_sell_in = df_sell_in.drop(columns=columnas_a_eliminar, errors='ignore')
        df_sell_out = df_sell_out.drop(columns=columnas_a_eliminar, errors='ignore')

        df_sell_in['SI/SO'] = 'SI'
        df_sell_out['SI/SO'] = 'SO'

        columnas_requeridas = ['Ultimo Eslabón', 'Canal 3']
        for col in columnas_requeridas:
            if col not in df_sell_in.columns:
                st.error(f"🚫 Columna faltante en Sell In: {col}")
                return
            if col not in df_sell_out.columns:
                st.error(f"🚫 Columna faltante en Sell Out: {col}")
                return

        # Limpieza Sell Out
        if 'Suma Outliers' in df_sell_out.columns:
            cols_out = df_sell_out.columns.tolist()
            cols_out.append(cols_out.pop(cols_out.index('Ultimo Eslabón')))
            cols_out.append(cols_out.pop(cols_out.index('Canal 3')))
            df_sell_out = df_sell_out[cols_out]
            df_sell_out = df_sell_out.loc[:, 'Suma Outliers':]
            cols_out = df_sell_out.columns.tolist()
            cols_out.insert(0, cols_out.pop(cols_out.index('Ultimo Eslabón')))
            cols_out.insert(1, cols_out.pop(cols_out.index('Canal 3')))
            df_sell_out = df_sell_out[cols_out]
            df_sell_out = df_sell_out.drop(columns=['Suma Outliers'])

        # Limpieza Sell In
        if 'Suma Outliers' in df_sell_in.columns:
            cols_in = df_sell_in.columns.tolist()
            cols_in.append(cols_in.pop(cols_in.index('Ultimo Eslabón')))
            cols_in.append(cols_in.pop(cols_in.index('Canal 3')))
            df_sell_in = df_sell_in[cols_in]
            df_sell_in = df_sell_in.loc[:, 'Suma Outliers':]
            cols_in = df_sell_in.columns.tolist()
            cols_in.insert(0, cols_in.pop(cols_in.index('Ultimo Eslabón')))
            cols_in.insert(1, cols_in.pop(cols_in.index('Canal 3')))
            df_sell_in = df_sell_in[cols_in]
            df_sell_in = df_sell_in.drop(columns=['Suma Outliers'])

        df_sell_in.columns = df_sell_out.columns
        df_concat = pd.concat([df_sell_in, df_sell_out])
        df_concat['ALERTA'] = 'Fin_Correc_Outliers'

        # ruta_stock = os.path.expanduser(
        #     "~/DERCO CHILE REPUESTOS SpA/Planificación y abastecimiento - Documentos/Planificación y Compras Ventas/Stock Historico Query.xlsx"
        # )
        # if not os.path.exists(ruta_stock):
        #     st.error(f"❌ Archivo de stock no encontrado: {ruta_stock}")
        #     return

        # st.success(f"📘 Archivo Stock Query cargado: {ruta_stock}")
        # df_stock = pd.read_excel(ruta_stock, sheet_name='Stock Query')
        archivo_stock = st.file_uploader("📤 Carga archivo Stock Query (.xlsx) si es muy pesado crea un nuevo archivo solo con la hoja Stock Query", type=["xlsx"])
        if archivo_stock is None:
            st.warning("⚠️ Esperando archivo Stock Query...")
            st.stop()
        df_stock = pd.read_excel(archivo_stock, sheet_name='Stock Query')
        st.success("✅ Archivo Stock Query cargado correctamente")

        df_stock['Ultimo Eslabón'] = df_stock['Ultimo Eslabón'].astype(str)

        df_siso = df_concat
        df_siso['Ultimo Eslabón'] = df_siso['Ultimo Eslabón'].astype(str)

        columnas_sumar = df_siso.columns[-20:-2]
        df_siso['Total'] = df_siso[columnas_sumar].sum(axis=1)
        df_siso['Venta Sku'] = df_siso.groupby('Ultimo Eslabón')['Total'].transform('sum')
        df_siso['%'] = df_siso.apply(lambda row: row['Total'] / row['Venta Sku'] if row['Venta Sku'] != 0 else 1, axis=1)

        st.dataframe(df_siso.head(20))
        st.success("✅ Proceso de corrección de disponibilidad avanzando...")


        df_stock['Fecha'] = pd.to_datetime(df_stock['Fecha'], errors='coerce')
        df_stock = df_stock.dropna(subset=['Fecha'])

        ultimas_24_fechas = df_stock['Fecha'].drop_duplicates().nlargest(24)
        df_stock = df_stock[df_stock['Fecha'].isin(ultimas_24_fechas)]

        st.info(f"📅 Fechas utilizadas: {ultimas_24_fechas.sort_values(ascending=False).dt.strftime('%Y-%m-%d').tolist()}")

        df_stock_pivot = df_stock.pivot(index='Ultimo Eslabón', columns='Fecha', values='Stock').reset_index()
        df_stock_pivot.columns = ['Ultimo Eslabón'] + [str(col)[:10] + '_stock' for col in df_stock_pivot.columns[1:]]

        df_siso_combined = df_siso.merge(df_stock_pivot, on='Ultimo Eslabón', how='left')
        df_siso_combined = df_siso_combined.fillna(0)

        st.success("✅ Archivos procesados y combinados correctamente.")
        st.dataframe(df_siso_combined.head(20))

        for column in df_stock_pivot.columns[1:]:
            df_siso_combined[column] = df_siso_combined[column] * df_siso_combined['%']

        col_start = df_siso_combined.columns.get_loc('Canal 3') + 1
        col_end = df_siso_combined.columns.get_loc('SI/SO')
        df_siso_combined['ALERTA 2'] = 'Fin_Cruce_Stock'

        cols_to_copy = df_siso_combined.columns[col_start:col_end]
        new_column_names = [col[:10] + '_Correc_Outliers' for col in cols_to_copy]
        new_columns = df_siso_combined[cols_to_copy].copy()
        new_columns.columns = new_column_names

        df_siso_combined = pd.concat([df_siso_combined, new_columns], axis=1)
        df_siso_combined['ALERTA 3'] = 'INI_CORRECCIÓN_DISPO'

        stock_cols = [col for col in df_siso_combined.columns if col.endswith('_stock')]
        correc_outliers_cols = [col for col in df_siso_combined.columns if col.endswith('_Correc_Outliers')]

        index_siso = df_siso_combined.columns.get_loc('SI/SO')
        cols_promedio = df_siso_combined.columns[index_siso - 18:index_siso]
        promedios = df_siso_combined[cols_promedio].mean(axis=1)

        result_cols = []
        for stock_col in stock_cols:
            correc_outliers_col = stock_col.replace('_stock', '_Correc_Outliers')
            if correc_outliers_col in correc_outliers_cols:
                nueva_columna = stock_col.replace('_stock', '_result')
                result_cols.append(nueva_columna)
                df_siso_combined[nueva_columna] = df_siso_combined.apply(
                    lambda row: promedios[row.name] if row[stock_col] < promedios[row.name] and row[correc_outliers_col] < promedios[row.name]
                    else row[correc_outliers_col], axis=1
                )

        cols_to_drop = [col for col in df_siso_combined.columns if col.endswith('_Correc_Outliers')][-24:]
        df_siso_combined.drop(columns=cols_to_drop, inplace=True)

        df_original = df_siso_combined.copy()
        columnas_despues = df_siso_combined.columns.tolist()
        columnas_antes = df_original.columns.tolist()
        columnas_eliminadas = [col for col in columnas_antes if col not in columnas_despues]
        columnas_quedaron = [col for col in columnas_despues if col in columnas_antes]

        st.write("🗃️ Columnas eliminadas:", columnas_eliminadas)
        st.write("📋 Columnas que quedaron:", columnas_quedaron)

        from datetime import datetime, timedelta
        import os

        meses_espanol_abreviado = {
            1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'
        }

        meses_espanol_completo = {
            1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
            7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
        }

        # usuario = os.getlogin()
        # fecha_actual = datetime.now()
        # fecha_siguiente = fecha_actual + timedelta(days=30)
        # fecha_anterior = fecha_actual - timedelta(days=30)

        # mes_siguiente = fecha_siguiente.month
        # mes_anterior = fecha_anterior.month

        # import os

        # usuario = os.path.expanduser("~")
        # anio_mes_guardar = fecha_actual.strftime(f"%Y-%m Ciclo {meses_espanol_abreviado[mes_siguiente]}-%y")
        # nombre_archivo_csv = fecha_actual.strftime(f"%Y.%m Correccion Disponibilidad {meses_espanol_completo[mes_anterior]}-%y") + ".csv"
        # nombre_archivo_parquet = nombre_archivo_csv.replace('.csv', '.parquet')

        # ruta_base_guardado = os.path.join(
        #     usuario,
        #     "derco chile repuestos spa",
        #     "Planificación y abastecimiento - Documentos",
        #     "Planificación y Compras Anastasia",
        #     "Carga Historia de Venta",
        #     anio_mes_guardar,
        #     "AFM",
        #     "Correccion Dispo"
        # )

        # os.makedirs(ruta_base_guardado, exist_ok=True)

        # ruta_guardar_csv = os.path.join(ruta_base_guardado, nombre_archivo_csv)
        # ruta_guardar_parquet = os.path.join(ruta_base_guardado, nombre_archivo_parquet)

        # df_siso_combined.to_csv(ruta_guardar_csv, sep=';', decimal=',', index=False)
        # df_siso_combined.to_parquet(ruta_guardar_parquet, index=False)

        # st.success("✅ Archivos guardados correctamente.")
        # st.write(f"💾 Ruta CSV: `{ruta_guardar_csv}`")
        # st.write(f"💾 Ruta Parquet: `{ruta_guardar_parquet}`")

        import io

        st.subheader("📤 Ahora debes guardar la corrección por disponibilidad")
        st.info("Esto permite tener un respaldo en formatos CSV y Parquet, por si necesitas revisar o auditar la información más adelante.")

        from datetime import datetime, timedelta

        meses_espanol_abreviado = {
            1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'
        }
        meses_espanol_completo = {
            1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
            7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
        }

        fecha_actual = datetime.now()
        fecha_siguiente = fecha_actual + timedelta(days=30)
        fecha_anterior = fecha_actual - timedelta(days=30)

        mes_siguiente = fecha_siguiente.month
        mes_anterior = fecha_anterior.month

        nombre_archivo_csv = fecha_actual.strftime(f"%Y.%m Correccion Disponibilidad {meses_espanol_completo[mes_anterior]}-%y") + ".csv"
        nombre_archivo_parquet = nombre_archivo_csv.replace('.csv', '.parquet')

        # Crear archivos en memoria
        csv_buffer = io.StringIO()
        df_siso_combined.to_csv(csv_buffer, sep=';', decimal=',', index=False)
        csv_data = csv_buffer.getvalue()

        parquet_buffer = io.BytesIO()
        df_siso_combined.to_parquet(parquet_buffer, index=False)
        parquet_data = parquet_buffer.getvalue()

        # Botones de descarga
        st.download_button(
            label="📥 Descargar CSV Corregido",
            data=csv_data,
            file_name=nombre_archivo_csv,
            mime="text/csv"
        )

        st.download_button(
            label="📥 Descargar Parquet Corregido",
            data=parquet_data,
            file_name=nombre_archivo_parquet,
            mime="application/octet-stream"
        )


        # import os
        # import pandas as pd
        # from datetime import datetime, timedelta

        # meses_espanol_abreviado = {1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'}
        # meses_espanol_completo = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}

        # usuario = os.getlogin()
        # fecha_actual = datetime.now()
        # fecha_siguiente = fecha_actual + timedelta(days=30)
        # fecha_anterior = fecha_actual - timedelta(days=30)

        # mes_siguiente = fecha_siguiente.month
        # mes_anterior = fecha_anterior.month

        # anio_mes_leer = fecha_actual.strftime(f"%Y-%m Ciclo {meses_espanol_abreviado[mes_siguiente]}-%y")
        # nombre_archivo = fecha_actual.strftime(f"%Y.%m Correccion Disponibilidad {meses_espanol_completo[mes_anterior]}-%y")
        # import os

        # usuario = os.path.expanduser("~")
        # ruta_leer = os.path.join(
        #     usuario,
        #     "derco chile repuestos spa",
        #     "Planificación y abastecimiento - Documentos",
        #     "Planificación y Compras Anastasia",
        #     "Carga Historia de Venta",
        #     anio_mes_leer,
        #     "AFM",
        #     "Correccion Dispo",
        #     f"{nombre_archivo}.parquet"
        # )

        # df_siso_combined = pd.read_parquet(ruta_leer)
        import streamlit as st
        import pandas as pd

        st.info("🚀 COMIENZO A DARLE FORMATO PARA PODER PREDECIR CON TOWERCAST")

        archivo_parquet = st.file_uploader("📂 Sube el archivo `.parquet` de corrección de disponibilidad que descargaste", type=["parquet"])

        if archivo_parquet is not None:
            df_siso_combined = pd.read_parquet(archivo_parquet)
            st.success("✅ Archivo parquet cargado correctamente.")
            st.dataframe(df_siso_combined.head())
        else:
            st.warning("⚠️ Debes subir un archivo .parquet para continuar.")
            st.stop()

        alerta2_index = df_siso_combined.columns.get_loc('ALERTA 2')
        selected_columns = df_siso_combined.columns[:2].tolist() + df_siso_combined.columns[alerta2_index+1:].tolist()
        df_filtered = df_siso_combined.loc[:, selected_columns]

        if 'ALERTA 3' in df_filtered.columns:
            df_filtered = df_filtered.drop(columns=['ALERTA 3'])

        df_melted = pd.melt(df_filtered, id_vars=df_filtered.columns[:2], var_name='Fecha', value_name='Venta')
        df_melted['Fecha'] = df_melted['Fecha'].str.replace('_Correc_Outliers', '').str.replace('_result', '')
        df_melted['ID transacción'] = ['TR' + str(i) for i in range(1, len(df_melted) + 1)]
        df_melted['idSKU_R3'] = ['SKU' + str(i) for i in range(1, len(df_melted) + 1)]

        cols = df_melted.columns.tolist()
        cols.insert(cols.index('Fecha') + 1, cols.pop(cols.index('ID transacción')))
        cols.insert(cols.index('Fecha') + 2, cols.pop(cols.index('idSKU_R3')))
        df_melted = df_melted[cols]

        df_melted.rename(columns={'Ultimo Eslabón': 'idSKU', 'Canal 3': 'Canal', 'Venta': 'Cantidad'}, inplace=True)
        df_melted['Unidad Medida'] = 'UN'
        df_melted['Pais'] = 'CL'
        df_melted['valor_trn'] = 0

        condiciones = [
            df_melted['Canal'].isin(['CL WALMART', 'CL EASY', 'CL SMU', 'CL TOTTUS', 'CL SODIMAC']),
            df_melted['Canal'].isin(['CL MAYORISTA', 'CL CES 01']),
            df_melted['Canal'].isin(['CL AUTOPLANET', 'CL AGROPLANET'])
        ]
        resultados = ['SO', 'SI', 'SO']
        df_melted['Tipo_Venta'] = np.select(condiciones, resultados, default='')

        df_melted['Cantidad'] = pd.to_numeric(df_melted['Cantidad'], errors='coerce').round()

        columnas_finales = [
            'Fecha', 'ID transacción', 'idSKU', 'idSKU_R3', 'Cantidad',
            'valor_trn', 'Canal', 'Unidad Medida', 'Pais', 'Tipo_Venta'
        ]
        df_melted = df_melted[columnas_finales]
        df_melted.rename(columns={'ID transacción': 'ID transaccion'}, inplace=True)

        df_si_so = df_melted[df_melted['Cantidad'] > 0].copy()
        df_si_so.reset_index(drop=True, inplace=True)
        df_si_so['idSKU_R3'] = 'SKU' + (df_si_so.index + 1).astype(str)
        df_si_so['ID transaccion'] = 'TR' + (df_si_so.index + 1).astype(str)

        from datetime import datetime
        import os

        usuario = os.path.expanduser("~")
        año_actual = datetime.now().year
        mes_actual = datetime.now().strftime('%m')

        archivo_mara = st.file_uploader("📤 Carga archivo MARA (.csv)", type=["csv"])
        if archivo_mara is None:
            st.warning("⚠️ Esperando archivo MARA...")
            st.stop()

        mara = pd.read_csv(archivo_mara, delimiter=';')
        st.success("✅ Archivo MARA cargado correctamente")


        # if archivos_mara:
        #     st.success(f"📄 Archivo MARA encontrado: {archivos_mara[0]}")
        #     mara = pd.read_csv(archivos_mara[0], delimiter=';')

        #     df_si_so = df_si_so.merge(
        #         mara[['Material_S4', 'Nombre Sector']],
        #         left_on='idSKU',
        #         right_on='Material_S4',
        #         how='left'
        #     ).drop(columns=['Material_S4'])

        #     sectores_filtrar = ['BAT', 'ACC', 'LUB', 'NEU', 'RALT', 'RMAQ']
        #     df_si_so = df_si_so[df_si_so['Nombre Sector'].isin(sectores_filtrar)].drop(columns=['Nombre Sector'])
        df_si_so = df_si_so.merge(
            mara[['Material_S4', 'Nombre Sector']],
            left_on='idSKU',
            right_on='Material_S4',
            how='left'
        ).drop(columns=['Material_S4'])

        sectores_filtrar = ['BAT', 'ACC', 'LUB', 'NEU', 'RALT', 'RMAQ']
        df_si_so = df_si_so[df_si_so['Nombre Sector'].isin(sectores_filtrar)].drop(columns=['Nombre Sector'])


        #     import os

        #     usuario = os.path.expanduser("~")
        #     nombre_archivosi_so = fecha_actual.strftime(f"SI_SO Ciclo {meses_espanol_abreviado[mes_siguiente]}-%Y")

        #     ruta_guardar_csvsi_so = os.path.join(
        #         usuario,
        #         "DERCO CHILE REPUESTOS SpA",
        #         "Planificación y abastecimiento - Documentos",
        #         "Planificación y Compras Anastasia",
        #         "Carga Historia de Venta",
        #         anio_mes_guardar,
        #         "AFM",
        #         "Correccion Dispo",
        #         f"{nombre_archivosi_so}_st.csv"
        #     )


        #     os.makedirs(os.path.dirname(ruta_guardar_csvsi_so), exist_ok=True)
        #     df_si_so.to_csv(ruta_guardar_csvsi_so, sep=',', index=False, encoding='utf-8')
        #     st.success("📁 Archivo final exportado correctamente:")
        #     st.code(ruta_guardar_csvsi_so)
        # else:
        #     st.warning(f"❌ No se encontró archivo MARA en: {ruta_mara}")
        import io

        nombre_archivosi_so = fecha_actual.strftime(f"SI_SO Ciclo {meses_espanol_abreviado[mes_siguiente]}-%Y")
        nombre_final_csv = f"{nombre_archivosi_so}_st.csv"

        csv_buffer = io.StringIO()
        df_si_so.to_csv(csv_buffer, sep=',', index=False, encoding='utf-8')
        csv_bytes = csv_buffer.getvalue().encode('utf-8')

        st.download_button(
            label="⬇️ Descargar CSV SI_SO",
            data=csv_bytes,
            file_name=nombre_final_csv,
            mime="text/csv"
        )


def generar_forecast():
    import pandas as pd
    import numpy as np
    import os
    from datetime import datetime
    from statsmodels.tsa.arima.model import ARIMA
    import streamlit as st
    import io
    import contextlib

    st.markdown("### 🔍 Selecciona el tipo de demanda para predecir")
    st.markdown("""
    <style>
    .stSelectbox div[data-baseweb="select"] > div {
        color: black !important;
    }
    </style>
""", unsafe_allow_html=True)

    cluster = st.selectbox("Tipo de Cluster", ["Todos", "Smooth", "Lumpy", "Erratic", "Intermittent"])

    archivo_subido = st.file_uploader("📁 Carga el archivo de ventas para predecir", type=["csv"])

    if archivo_subido is None:
        return

    df = pd.read_csv(archivo_subido, delimiter=',', decimal=',', encoding='utf-8')
    df = df.rename(columns={'Canal': 'Canal 3', 'Cantidad': 'Venta', 'idSKU': 'Ultimo Eslabón'})
    df['Venta'] = pd.to_numeric(df['Venta'], errors='coerce')
    df = df.dropna(subset=['Venta'])

    df['ID'] = df['Ultimo Eslabón'].astype(str) + df['Canal 3'].astype(str)
    df['Fecha'] = pd.to_datetime(df['Fecha'], format='%Y-%m-%d')
    df = df.sort_values(by=['Ultimo Eslabón', 'Fecha']).reset_index(drop=True)
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
    fecha_default = (datetime.today() + relativedelta(months=1)).replace(day=1)
    fecha_seleccionada = st.date_input("📅 Selecciona el mes de inicio del forecast", value=fecha_default)
    fecha_inicio_forecast = fecha_seleccionada.replace(day=1)

    with st.spinner("🔮 Generando Forecast..."):

        pivot_table = df.pivot_table(index=['Ultimo Eslabón', 'Canal 3'], columns='Fecha', values='Venta', fill_value=0).reset_index()
        pivot_table = pivot_table[pivot_table.iloc[:, 2:].sum(axis=1) >= 2]

        ultima_fecha = pivot_table.columns[-1]
        primer_fecha = ultima_fecha - pd.DateOffset(months=23)
        ventas_ultimos_24_meses = pivot_table.loc[:, primer_fecha:ultima_fecha]

        intervalos = ventas_ultimos_24_meses.apply(lambda x: (x != 0).astype(int).diff().fillna(1).abs().sum(), axis=1)
        ventas_activas = (ventas_ultimos_24_meses != 0).sum(axis=1)
        pivot_table['ADI'] = intervalos / ventas_activas
        ventas_no_cero = ventas_ultimos_24_meses.replace(0, np.nan)
        pivot_table['CV²'] = (ventas_no_cero.std(axis=1) / ventas_no_cero.mean(axis=1)) ** 2

        def clasificar_demanda(row):
            if row['ADI'] < 1.32 and row['CV²'] < 0.49: return 'Smooth'
            elif row['ADI'] >= 1.32 and row['CV²'] < 0.49: return 'Intermittent'
            elif row['ADI'] < 1.32 and row['CV²'] >= 0.49: return 'Erratic'
            else: return 'Lumpy'

        pivot_table['Demand Type'] = pivot_table.apply(clasificar_demanda, axis=1)

        melted_data = pivot_table.melt(
            id_vars=['Ultimo Eslabón', 'Canal 3', 'ADI', 'CV²', 'Demand Type'],
            var_name='Fecha',
            value_name='Venta'
        )
        melted_data['Fecha'] = pd.to_datetime(melted_data['Fecha'], format='%Y-%m-%d')
        melted_data['ID'] = melted_data.apply(lambda row: f"{row['Ultimo Eslabón']}_{row['Canal 3']}", axis=1)

        if cluster != "Todos":
            melted_data = melted_data[melted_data['Demand Type'] == cluster]

        data = melted_data[['ID', 'Fecha', 'Venta']].copy()
        predictions = []
        log_buffer = io.StringIO()

        for id_value in data['ID'].unique():
            data_id = data[data['ID'] == id_value]
            data_id.set_index('Fecha', inplace=True, drop=True)
            data_id.index.freq = 'MS'
            y = data_id['Venta']

            if y.isnull().any() or np.isinf(y).any():
                log_buffer.write(f"⚠️ Serie con nulos o infinitos: {id_value}\n")
                continue

            try:
                with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
                    model = ARIMA(y, order=(5, 1, 0))
                    model_fit = model.fit()
                    future_predictions = model_fit.forecast(steps=12)


                    future_dates = pd.date_range(start=fecha_inicio_forecast, periods=12, freq='MS').strftime('%Y-%m-%d')

                    for date, pred in zip(future_dates, future_predictions):
                        predictions.append({
                            'ID': id_value,
                            'Fecha': date,
                            'Prediccion_Venta': pred
                        })
            except Exception as e:
                log_buffer.write(f"❌ Error al modelar {id_value}: {e}\n")
                continue

        predictions_df = pd.DataFrame(predictions)
        predictions_df['ID'] = predictions_df['ID'].str.replace('_', '', regex=True)
        current_month = datetime.now().strftime('%Y-%m')
        cluster_suffix = cluster if cluster != "Todos" else "All"

        output_filename = f"TOWERZ_PREDICTIONS_{current_month}_{cluster_suffix}.xlsx"
        buffer = io.BytesIO()
        predictions_df.to_excel(buffer, index=False, sheet_name='Predicciones')
        buffer.seek(0)

        st.success("✅ Forecast generado con éxito.")

        # Guardamos el botón en una variable
        downloaded = st.download_button(
            label="📥 Descargar Forecast",
            data=buffer,
            file_name=output_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Solo mostrar animación si el usuario descargó
        if downloaded:
            st.snow()
            st.session_state.forecast_ready = False





import streamlit as st
from PIL import Image
from datetime import datetime
import time

# Bloque para expirar la app después del 18-05-2025
# fecha_limite = datetime.strptime("2025-08-01", "%Y-%m-%d")
# if datetime.now() > fecha_limite:
#     st.error("⛔ Esta aplicación ha expirado. Contacta a soporte para renovarla.")
#     st.stop()

if 'forecast_ready' not in st.session_state:
    st.session_state.forecast_ready = False



st.set_page_config(
    page_title="TowerCast | Corrección y Predicción de Ventas",
    page_icon="📈",
    layout="wide"
)

import os
import sys
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return encoded
ruta_fondo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fondo_towercast_tech_dark.png")
img_base64 = get_base64_image(ruta_fondo)

st.markdown(f"""
    <style>
    html, body, [data-testid="stAppViewContainer"] {{
        background: url("data:image/png;base64,{img_base64}") no-repeat center center fixed;
        background-size: cover;
        color: white !important;
    }}

    [data-testid="stHeader"] {{
        background-color: #0B1A2A !important;
        color: white !important;
    }}

    div.stButton > button {{
        border: 1px solid #2E86C1;
        color: white;
        background-color: #2E86C1;
        transition: all 0.3s ease;
        font-weight: bold;
        border-radius: 10px;
        height: 3em;
        font-size: 16px;
    }}

    div.stButton > button:hover {{
        background-color: #1B4F72;
        color: #fff;
        transform: scale(1.03);
    }}

    .fade-in {{
        animation: fadeIn 1.5s ease-in-out;
    }}

    @keyframes fadeIn {{
        0% {{ opacity: 0; transform: translateY(-20px); }}
        100% {{ opacity: 1; transform: translateY(0); }}
    }}

    .block-container {{
        padding-top: 20px;
        background-color: rgba(0,0,0,0.4);
        border-radius: 10px;
        padding: 2rem;
    }}

    h1, h2, h3, h4, h5, h6, p, div {{
        color: white !important;
    }}
    </style>
""", unsafe_allow_html=True)





# Logo a la izquierda y título centrado
col_logo, col_empty, col_title, col_empty2, col_right = st.columns([1.2, 0.5, 4, 0.5, 1])
import sys
import os

with col_logo:
    if getattr(sys, 'frozen', False):
        ruta_logo = os.path.join(sys._MEIPASS, "logo_towercast.png")
    else:
        ruta_logo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo_towercast.png")

    logo = Image.open(ruta_logo)
    st.image(logo, width=200)


with col_title:
    st.markdown("""
        <div class='fade-in' style='text-align: center; padding-top: 30px;'>
            <h1 style='color: #2E86C1; font-size: 36px;'>📊 Corrección y Predicción de Ventas</h1>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='margin-top:10px;margin-bottom:40px;'>", unsafe_allow_html=True)

# Botones funcionales
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🧮 Corregir Sell In", use_container_width=True):
        with st.spinner("🔄 Procesando Sell In..."):
            time.sleep(1.2)
            corregir_sellin()
        st.success("✅ Sell In corregido con éxito.")
        st.balloons()

with col2:
    if st.button("📊 Corregir Sell Out", use_container_width=True):
        with st.spinner("🔄 Procesando Sell Out..."):
            time.sleep(1.2)
            corregir_sellout()
        st.success("✅ Sell Out corregido con éxito.")
        st.balloons()

with col3:
    if st.button("🔍 Corregir Disponibilidad", use_container_width=True):
        with st.spinner("🔄 Corrigiendo Disponibilidad..."):
            time.sleep(1.2)
            corregir_disponibilidad()
        st.success("✅ Disponibilidad corregida.")
        st.snow()

with col4:
    if st.button("📈 Generar Forecast", use_container_width=True):
        st.session_state.forecast_ready = True

if st.session_state.forecast_ready:
    generar_forecast()




# Footer
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"""
    <div style='text-align: center; font-size: 13px; color: #888888;'>
        Última ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | TowerCast by Enrique Torres © 2025
    </div>
""", unsafe_allow_html=True)

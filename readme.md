  <h1> SeguridadAlimentaria </h1>
  <p>Este repositorio contiene diferentes datos abiertos de diferentes fuentes para calcular hacer un análisis de la
    seguridad alimentaria en el departamento
    del Cauca, está dividio en diferentes carpetas
  </p>
  <ol>
    <li> Censo Nacional Agropecuario/: Aquí hay principalmente 1 Archivo y una carpeta: </li>
    <ul>
      <li> Datos/ </li>
      <ul>
        <li> *.csv: </li> 11 archivos en formato csv que contiene la información recaudada por por el <a
          href="http://microdatos.dane.gov.co/index.php/catalog/513/get_microdata"> censo nacional agropecuario del año
          2014</a>
        <li>TEMATICA_DISENO DE REGISTRO CNA2014.xlsx </li> Diccionario que explica el significado de los campos en las
        tablas del censo nacional agropecuario
      </ul>
      <li> CensoAgro.ipynb: </li> Archivo de Jupyter en el cual se leen los csv del censo nacional agropecuario y se
      filtran los datos necesarios para el cálculo de la seguridad alimentaria
    </ul>
    <li> Datos Sociodemograficos/ :</li>
    <ul>
      <li> DANE_Sociodemograficos.csv: </li> Archivo que contiene los datos sociodemograficos de los diferentes
      municipios de Colombia <a
        href="https://github.com/MITCriticalData-Colombia/Dengue-MetaData/blob/main/DANE_Dengue_Data_Variables.csv">
        Fuente </a>
    </ul>
    <li> ENSIN/: </li>
    <ul>
      <li> Datos/ </li>
      <ul>
        <li> *.csv: </li> 8 archivos en formato csv que contiene la información recaudada por la <a
          href="https://www.icbf.gov.co/bienestar/nutricion/encuesta-nacional-situacion-nutricional"> encuesta nacional
          de la situación nutricional (ENSIN) en el año 2015</a>
        <li> Dic_Ensin_Pública.xlsx :</li> Diccionario con el significado de los campos de cada tabla y sus relaciones.
      </ul>
      <li> ENSIN.ipynb :</li> Archivo de Jupyter en el cual se leen los csv de la ENSIN y se filtran los datos
      necesarios para el cálculo de la seguridad alimentaria
    </ul>
    <li> Medida De Pobreza Multidimensional (MPM) - Dane/: </li>
    <ul>
      <li>Data/</li>
      <ul>
        <li> MPM2018_shapefile/ </li> Contiene el folder con los archivos necesarios para leer el shapefile que contiene
        los datos de la <a
          href="https://www.dane.gov.co/index.php/estadisticas-por-tema/pobreza-y-condiciones-de-vida/pobreza-y-desigualdad/medida-de-pobreza-multidimensional-de-fuente-censal">
          medida de pobreza multidimensional del dane para cada municipio en Colombia en 2018 </a>
        <li> Diccionario_MPM.xlsx </li> Diccionario que explica el significado de cada uno de los campos de la encuesta
        MPM
        <li> DIVIPOLA_Municipios.csv </li> Archivo csv que contiene información de cada municipio y departamento como
        latitud, longitud, código, entre otras.
      </ul>
      <li>Datos_Socioecnomicos(resumen).ipynb :</li> En este archivo se puede ver una descripción de lo que tiene la
      carpeta MPM2018_shapefile/ y la estructura de los datos.
      <li> MPM_extraction.ipynb :</li> Este es el archivo utilizado para la extracción de los datos en formato shapefile
      a csv
      <li> MPM_Output/ </li> Directorio que contiene el archivo mpm_data_dane_2018.csv resultante del notebook
      MPM_extraction.ipynb
    </ul>
    <li> Satelital Data/: </li> Esta carepta esta dividida en 2 secciones:
    <ol>
      <li> WorldClim Data/ :</li> Esta carpeta contiene los archivos convertidos a csv extraidos <a
        href="https://github.com/MITCriticalData-Colombia/Dengue-MetaData/blob/Dengue_Temperature_Precipitation%2B%2B/WorldClimTemperature2007_2018.ipynb">(Código
        temperatura) </a> de <a href="https://www.worldclim.org/data/monthlywth.html"> WorldClim en formato .tif </a>
      <ul>
        <li> elevation.csv </li> Elevación para cada municipio de Colombia
        <li> precipitation2007_2018.csv </li> Precipitación para cada municipio de Colombia mensual entre 2007 y 2018
        <li> tempearture2007-2018.csv </li> Temperatura para cada municipio de Colombia mensual entre 2007 y 2018
      </ul>
      <li> GoogleEarthEngine/ </li> Esta carpeta interactua con la plataforma de google <a
        href="https://earthengine.google.com/"> Google Earth Engine </a> para obtener tanto caracterizticas de satelites
      como datos de temperatura y precipitación en areas especificas como descargar imagenes satelitales en formato .tif
      contiene:
      <ul>
        <li> GEE_RemoteSensing.ipynb : Archivo que muestra como hacer uso de la API de Python de GEE para hacer senso
          remoto y ver datos de Popayan de temperatura, precipitación e imagenes satelitales.</li>
        <img width="500" alt="Captura de Pantalla 2021-07-29 a la(s) 7 04 08 p  m"
          src="https://user-images.githubusercontent.com/57241831/127581950-4c45145f-3ebd-4549-b27f-00d50bea320d.png">
        <li> GEE_Time_Series_Download_NoClouds.ipynb : Archivo que obtiene una imagen del satelite landsat 8 con las
          bandas RGB (Rojo, verde y Azul) en formato .tif de Popayan para cada año entre 2013 y 2020, aplicando un
          algoritmo de eliminación de nubes.</li>
        <img width="500" alt="Captura de Pantalla 2021-07-29 a la(s) 7 07 32 p  m"
          src="https://user-images.githubusercontent.com/57241831/127582281-0d5fa094-5244-4f4c-a6ae-a5e28ef569e0.png">
        <li> Images/ </li> Contiene las imagenes satelitales resultantes de GEE_Time_Series_Download_NoClouds.ipynb
        <li> Nota: </li> Adicionalmente se ha explorado la API de JavaScript de Google Earth Engine con ejemplo como
        <ul>
          <li> <a href="https://code.earthengine.google.com/2661096f995f566e15ca544a11df958c"> Imagen satelital de
              Popayán sin nubes </a> : </li>
          <img width="500" alt="Captura de Pantalla 2021-07-29 a la(s) 12 46 25 p  m"
            src="https://user-images.githubusercontent.com/57241831/127543116-63d1d11f-f6f0-425f-871b-06358c6fef2f.png">
          <li> <a href="https://code.earthengine.google.com/958ca799e04b979cb6e729f1bc266edd"> Temperatura de Popayán
              diaria </a> : </li>
          <img width="500" alt="Captura de Pantalla 2021-07-29 a la(s) 12 48 00 p  m"
            src="https://user-images.githubusercontent.com/57241831/127543314-c2c0ac7d-773c-4082-a8ca-e7f9f162d990.png">
        </ul>
      </ul>
    </ol>
    <li> Sivigilia/: </li>
    <ul>
      <li>Data/ </li>
      <ul>
        <li> DIVIPOLA_Municipios.csv :</li> Archivo con información basica de los municipios como nombre, departamento,
        y código
        <li> *.csv y *.xlsx </li> Archivos del <a
          href="https://www.ins.gov.co/Direcciones/Vigilancia/Paginas/SIVIGILA.aspx"> Sistema Nacional de Vigilancia en
          Salud Pública -SIVIGILA </a> que contienen los ingresos hospitalarios para los años entre 2007 y 2019 en
        Colombia reportados por municipio
      </ul>
      <li>sivigila(Resumen).ipynb : Archivo que hace un análisis de los datos de sivigila para entender su contenido
      </li>
      <img width="500" alt="Captura de Pantalla 2021-07-29 a la(s) 1 58 21 p  m"
        src="https://user-images.githubusercontent.com/57241831/127550414-5611b644-af37-41e8-af95-354697fdf667.png">
    </ul>
    <li> joinAllDataframes.ipynb: </li> Archivo que lee los datos Sociodemograficos, de Sivigila, Socioeconónicos y
    Climáticos de WorldClim y los junta en un solo dataset
    <li> OutputDataset/: </li> Carepta que contiene el archivo SeguridadAlimentariaV1.csv el cual es el dataset
    resultante de joinAllDataframes.ipynb
    <li>FoodSecurityIndex/: </li>
    <ul>
      <li>Data_Countries/ </li> Archivo que contiene los dataset con los pesos para el calculo del <a
        href="https://foodsecurityindex.eiu.com/Index"> food security index </a> por pais
      <li> LinearRegression_FoodSecurityIndex.ipynb :</li> Regresión lineal del food security index y los coeficientes
      para futura imputación de datos
    </ul>
  </ol>
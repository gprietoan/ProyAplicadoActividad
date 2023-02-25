#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importacion de librerias para graficas y manejo de datos
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

data = pd.read_csv('Programa_social_FA.csv')
data.shape


# In[2]:


#se realiza un proceso de reemplazo de valores no especificados de las columnas por un valores NaN
data['Genero'] = data['Genero'].apply(lambda x: x.replace('ND','NaN') if 'ND' in str(x) else x)
data['TipoBeneficio'] = data['TipoBeneficio'].apply(lambda x: x.replace('ND','NaN') if 'ND' in str(x) else x)
data['TipoDocumento'] = data['TipoDocumento'].apply(lambda x: x.replace('No Definido','NaN') if 'ND' in str(x) else x)
data['Discapacidad'] = data['Discapacidad'].apply(lambda x: x.replace('ND','NaN') if 'ND' in str(x) else x)
data['RangoEdad'] = data['RangoEdad'].apply(lambda x: x.replace('-',' a ') if '-' in str(x) else x)


# In[3]:


data_bogota = data[data['CodigoMunicipioAtencion'] == 11001]
data_bogota.shape


# In[4]:


data.dtypes


# In[5]:


#Se determina que casi no se encuentran variables numericas para realizar el analisis 
data_bogota.describe()


# In[6]:


#invocar el metodo figure_factory para enviar parametros fuera de linea
import plotly.figure_factory as ff
# desactivar los warning
import warnings
warnings.filterwarnings('ignore')


# In[7]:


#filtramos dataframe para no tener datos NaN en la columna Genero
data_bogota['Etnia'].value_counts().sort_values(ascending = True)


# In[8]:


data_bogota.head()


# In[9]:


import plotly 
plotly.offline.init_notebook_mode(connected = True)

#invocar el metodo figure_factory para enviar parametros fuera de linea
import plotly.figure_factory as ff
# desactivar los warning
import warnings
warnings.filterwarnings('ignore')


# In[10]:


import plotly.graph_objs as go
import numpy as np
from plotly.subplots import make_subplots

data_bogota_filtrado1 = data_bogota[data_bogota.Discapacidad == 'SI']
data_bogota_filtrado1 = data_bogota[data_bogota.CantidadDeBeneficiarios > 1000]

trace0 = go.Box(y = np.log10(data_bogota_filtrado1['CantidadDeBeneficiarios'][data_bogota_filtrado1.Genero == 'Hombre']),
                name = 'Genero_Hombre',
                marker = dict(color = 'rgb(25,25,112)',)
        )
trace1 = go.Box(y = np.log10(data_bogota_filtrado1['CantidadDeBeneficiarios'][data_bogota_filtrado1.Genero == "Mujer"]),
                name = 'Genero_Mujer',
                marker = dict(color = 'rgb(255,105,180)',)
               )

layout = go.Layout(
                    title = 'Numero de beneficiarios del programa por Genero',
                    yaxis = {'title': 'Numero de benficiarios escaladas a logaritmos'}
                )
data_g = [trace0,trace1]
plotly.offline.iplot({'data':data_g, 'layout':layout})


# In[11]:


data_bogota_filtrado2 = data_bogota[data_bogota.EstadoBeneficiario == 'ACTIVO']
data_bogota_filtrado2 = data_bogota_filtrado2[data_bogota_filtrado2.CantidadDeBeneficiarios > 100]
data_bogota_filtrado2 = data_bogota_filtrado2[data_bogota_filtrado2.TipoDocumento != 'NaN']
data_bogota_filtrado2.shape


# In[12]:


trace0 = go.Box(y = np.log10(data_bogota_filtrado2['CantidadDeBeneficiarios'][data_bogota_filtrado2.TipoDocumento == 'CC']),
                name = 'Cedula',
                marker = dict(color = 'rgb(25,25,112)',)
        )
trace1 = go.Box(y = np.log10(data_bogota_filtrado2['CantidadDeBeneficiarios'][data_bogota_filtrado2.TipoDocumento =="CE"]),
                name = 'Cedula Extranjeria',
                marker = dict(color = 'rgb(255,0,128)',)
               )
trace2 = go.Box(y = np.log10(data_bogota_filtrado2['CantidadDeBeneficiarios'][data_bogota_filtrado2.TipoDocumento =="TI"]),
                name = 'Tarjeta',
                marker = dict(color = 'rgb(75,0,130)',)
               )
trace3 = go.Box(y = np.log10(data_bogota_filtrado2['CantidadDeBeneficiarios'][data_bogota_filtrado2.TipoDocumento =="RC"]),
                name = 'Registro Civil',
                marker = dict(color = 'rgb(112,128,144)',)
               )
layout = go.Layout(
                    title = 'Numero de beneficiarios del programa por Tipo de documento',
                    yaxis = {'title': 'Numero de benficiarios escaladas a logaritmos'}
                )
data_g = [trace0,trace1,trace2,trace3]
plotly.offline.iplot({'data':data_g, 'layout':layout})


# In[13]:


data_bogota_filtrado3 = data_bogota[data_bogota.EstadoBeneficiario == 'ACTIVO']
data_bogota_filtrado3 = data_bogota_filtrado3[data_bogota_filtrado3.CantidadDeBeneficiarios > 100]
data_bogota_filtrado3 = data_bogota_filtrado3[data_bogota_filtrado3.TipoPoblacion != 'NaN']
data_bogota_filtrado3.shape


# In[14]:


data_bogota_filtrado3['TipoPoblacion'].value_counts().sort_values(ascending = True)


# In[15]:


trace0 = go.Box(y = np.log10(data_bogota_filtrado3['CantidadDeBeneficiarios'][data_bogota_filtrado3.TipoPoblacion == 'UNIDOS']),
                name = 'UNIDOS',
                marker = dict(color = 'rgb(25,25,112)',)
        )
trace1 = go.Box(y = np.log10(data_bogota_filtrado3['CantidadDeBeneficiarios'][data_bogota_filtrado3.TipoPoblacion =="SISBEN"]),
                name = 'SISBEN',
                marker = dict(color = 'rgb(255,0,128)',)
               )
trace2 = go.Box(y = np.log10(data_bogota_filtrado3['CantidadDeBeneficiarios'][data_bogota_filtrado3.TipoPoblacion =="DESPLAZADOS"]),
                name = 'DESPLAZADOS',
                marker = dict(color = 'rgb(75,0,130)',)
               )
layout = go.Layout(
                    title = 'Numero de beneficiarios del programa por Tipo de Poblacion',
                    yaxis = {'title': 'Numero de benficiarios escaladas a logaritmos'}
                )
data_g = [trace0,trace1,trace2]
plotly.offline.iplot({'data':data_g, 'layout':layout})


# In[17]:


#creacion de dataframe con ciertas columnsa categoricas para convertirlas en variables numericas
data_bogota_filtrado = data[["Genero","CantidadDeBeneficiarios","RangoEdad","EstadoBeneficiario","CodigoMunicipioAtencion"]]
data_bogota_filtrado = data_bogota_filtrado[data_bogota_filtrado["CodigoMunicipioAtencion"] == 11001]
data_bogota_filtrado


# In[18]:


temp_df = data_bogota_filtrado[data_bogota_filtrado.EstadoBeneficiario == 'ACTIVO']
temp_df = temp_df[temp_df.CantidadDeBeneficiarios > 100]
temp_df.shape


# In[19]:


data0 = [{
    'x': temp_df['RangoEdad'],
    'type': 'scatter',
    'y': temp_df['CantidadDeBeneficiarios'],
    'mode':'markers',
    'text': temp_df['CantidadDeBeneficiarios']
} for t in set (temp_df.RangoEdad)]

layout = {'title': 'Cruce entre el RangoEdad vs Cantidad de Beneficiarios',
          'xaxis': {'title': 'RangoEdad'},
          'yaxis' :{'title': 'Cantidad Beneficiarios'},
          'plot_bgcolor': 'rgb(204,229,255)'
         }

plotly.offline.iplot({'data':data0, 'layout': layout})


# In[141]:


import seaborn as sns
data_bogota_corr = data_bogota[["Genero","TipoPoblacion","TipoDocumento","CantidadDeBeneficiarios"]]
data_bogota_corr


# In[142]:


data_bogota_corr = data_bogota_corr[data_bogota_corr.CantidadDeBeneficiarios != "NaN"]
data_bogota_corr = data_bogota_corr[data_bogota_corr.Genero != 'NaN']
data_bogota_corr = data_bogota_corr[data_bogota_corr.TipoDocumento != 'No Definido']
data_bogota_corr = data_bogota_corr[data_bogota_corr.TipoPoblacion != 'SIN ESPECIFICAR']
data_bogota_corr = data_bogota_corr[data_bogota_corr.TipoPoblacion != 'ND']
data_bogota_corr.shape


# In[143]:


data_bogota_corr.head()


# In[144]:


def create_cambiar(df, var_name):
    cambio = pd.get_dummies(df[var_name], prefix = var_name)
    df = df.drop([var_name],axis = 1)
    df = pd.concat([df,cambio],axis = 1)
    return df


# In[145]:


data_bogota_corr = create_cambiar(data_bogota_corr,"Genero")
data_bogota_corr = create_cambiar(data_bogota_corr,"TipoPoblacion")
data_bogota_corr = create_cambiar(data_bogota_corr,"TipoDocumento")
data_bogota_corr


# In[146]:


from matplotlib import pyplot as plt
plt.style.use('ggplot')
sns.set(rc={'figure.figsize':(25,15)})
color = sns.color_palette()
corrmat = data_bogota_corr.corr(method = 'pearson')
p = sns.heatmap(corrmat, annot = True, cmap = sns.diverging_palette(202,0,as_cmap = True))


# In[150]:


import seaborn as sns
data_bogota_torta1 = data_bogota[["Genero","TipoPoblacion","TipoDocumento","CantidadDeBeneficiarios","Etnia"]]
data_bogota_torta1 = data_bogota_torta1[data_bogota_torta1.TipoPoblacion != 'SIN ESPECIFICAR']
data_bogota_torta1 = data_bogota_torta1[data_bogota_torta1.TipoPoblacion != 'ND']
data_bogota_torta1


# In[151]:


#grafica de torta de tipo pie, con go
#agrupar los datos por categoria y los vamos a ordenar
import plotly.graph_objs as go

numero_apps_categoria = data_bogota_torta1['TipoPoblacion'].value_counts().sort_values(ascending = True)
data_t1 = [go.Pie(labels = numero_apps_categoria.index, values = numero_apps_categoria.values, hoverinfo = 'label+value')]
plotly.offline.iplot(data_t1,  filename = 'active_Category')


# In[177]:


import seaborn as sns
data_bogota_torta2 = data[["NombreMunicipioAtencion","TipoPoblacion","TipoDocumento","CantidadDeBeneficiarios","TipoBeneficio","EstadoBeneficiario"]]
data_bogota_torta2 = data_bogota_torta2[data_bogota_torta2.EstadoBeneficiario == 'ACTIVO']
data_bogota_torta2 = data_bogota_torta2[data_bogota_torta2.TipoBeneficio != 'ND']
data_bogota_torta2 = data_bogota_torta2[data_bogota_torta2.TipoBeneficio != 'NaN']
data_bogota_torta2


# In[178]:


import plotly.graph_objs as go
import numpy as np
import matplotlib.pyplot as plt

numero_apps_categoria = data_bogota_torta2['TipoBeneficio'].value_counts().sort_values(ascending = True)
data_g = [go.Pie(labels = numero_apps_categoria.index, values = numero_apps_categoria.values, hoverinfo = 'label+value')]
plotly.offline.iplot(data_g,  filename = 'active_Category')


# In[167]:


import matplotlib.pyplot as plt
import seaborn as sns
data_histo1 = data[["Genero","TipoPoblacion","TipoDocumento","CantidadDeBeneficiarios","Etnia"]]
data_histo1 = data_histo1[data_histo1.TipoPoblacion != 'SIN ESPECIFICAR']
data_histo1 = data_histo1[data_histo1.TipoPoblacion != 'ND']
data_histo1


# In[203]:


#histograamas de frecuencias para distribuciones normal

plt.hist(data_histo1["TipoPoblacion"],bins=10,color='#F2AB6D')
plt.title("Histograma de beneficiarios")
plt.xlabel("Tipo de Población en el programa")
plt.ylabel("Frecuencia")


# In[174]:


import matplotlib.pyplot as plt
import seaborn as sns
data_histo2 = data[["Genero","TipoPoblacion","TipoDocumento","CantidadDeBeneficiarios","Etnia","RangoEdad","TipoBeneficio"]]
data_histo2 = data_histo2[data_histo2.TipoPoblacion != 'SIN ESPECIFICAR']
data_histo2 = data_histo2[data_histo2.TipoPoblacion != 'ND']
data_histo2 = data_histo2[data_histo2.Etnia != 'ND']
data_histo2 = data_histo2[data_histo2.Etnia != 'NaN']
data_histo2 = data_histo2[data_histo2.TipoBeneficio != 'NaN']
data_histo2


# In[213]:


plt.hist(data_histo2["RangoEdad"],color='#F2AB6D',rwidth=0.85)
plt.title("Histograma de beneficiarios")
plt.xlabel("Rando de edad en el programa")
plt.ylabel("Frecuencia")


# In[195]:


import matplotlib.pyplot as plt
import seaborn as sns
data_histo3 = data[["NivelEscolaridad"]]
data_histo3 = data_histo3[data_histo3["NivelEscolaridad"] != 'ND']
data_histo3


# In[220]:


plt.hist(data_histo3["NivelEscolaridad"],bins=20,color='#F2AB6D',rwidth=0.85)
plt.title("Histograma de beneficiarios")
plt.xlabel("Nivel de escolaridad en el programa")
plt.ylabel("Frecuencia")


# In[237]:


import matplotlib.pyplot as plt
import seaborn as sns
data_histo4 = data[["RangoBeneficioConsolidadoAsignado"]]
data_histo4 = data_histo4[data_histo4["RangoBeneficioConsolidadoAsignado"] != 'NaN']
data_histo4 = data_histo4[data_histo4["RangoBeneficioConsolidadoAsignado"] != 'ND']
data_histo4


# In[238]:


#filtramos dataframe para no tener datos NaN en la columna Genero
data_histo4['RangoBeneficioConsolidadoAsignado'].value_counts().sort_values(ascending = True)


# In[240]:


plt.hist(data_histo4["RangoBeneficioConsolidadoAsignado"],bins=5,color='#F2AB6D',rwidth=0.85)
plt.title("Histograma de beneficiarios")
plt.xlabel("Rango de beneficio consolidado")
plt.ylabel("Frecuencia")


# In[241]:


import matplotlib.pyplot as plt
import seaborn as sns
data_histo5 = data[["Titular"]]
data_histo5 = data_histo5[data_histo5["Titular"] != 'NaN']
data_histo5 = data_histo5[data_histo5["Titular"] != 'ND']
data_histo5


# In[244]:


plt.hist(data_histo5["Titular"],bins=20,color='#F2AB6D')
plt.title("Histograma de beneficiarios")
plt.xlabel("Rango de beneficio consolidado")
plt.ylabel("Frecuencia")


# In[37]:


data_vio1 = data[["CodigoMunicipioAtencion","Genero","TipoPoblacion","TipoDocumento","CantidadDeBeneficiarios","Etnia","RangoEdad","TipoBeneficio"]]
#data_vio1 = data_vio1[data_vio1.CodigoMunicipioAtencion == 11001]
data_vio1 = data_vio1[data_vio1.CantidadDeBeneficiarios > 10]
data_vio1 = data_vio1[data_vio1.TipoPoblacion != 'SIN ESPECIFICAR']
data_vio1 = data_vio1[data_vio1.TipoPoblacion != 'ND']
data_vio1 = data_vio1[data_vio1.Etnia != 'ND']
data_vio1 = data_vio1[data_vio1.Etnia != 'NaN']
data_vio1 = data_vio1[data_vio1.TipoBeneficio != 'NaN']
data_vio1


# In[38]:


data_vio1['TipoBeneficio'].value_counts().sort_values(ascending = True)


# In[39]:


from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt


# In[40]:


groups = data_vio1.groupby('TipoBeneficio').filter(lambda x: len(x) > 10).reset_index()
print('Promedio de los datos no vacios CantidadDeBeneficiarios = ',np.nanmean(list(groups.CantidadDeBeneficiarios)))


# In[36]:


import plotly.graph_objects as go
#parametro hsl define el tipo de graficacion
#linespace tamaño de la pantalla
#shapes 
C = ['hsl('+str(h)+', 50%'+'50%)' for h in np.linspace(0,720,len(set(groups.CantidadDeBeneficiarios)))]
#creacion de variable tipo figura
fig = go.Figure(
    #contenedor de elementos graficos
    #tickangle parametros de inclinacion del texto
    layout = {'title': 'Cantidad de benficiarios por cada tipo de Benficio',
              'xaxis': {'tickangle': -40},
              'yaxis': {'title': 'CantidadDeBeneficiarios'},
              'plot_bgcolor': 'rgb(230,230,230)',
              'shapes':[{'type': 'line',
                         'x0': -2,
                         'y0': np.nanmean(list(groups.CantidadDeBeneficiarios)),
                         'x1': 18,
                         'y1':np.nanmean(list(groups.CantidadDeBeneficiarios)),
                         'line':{'dash':'dashdot'}
                        }]
             },
    data = [{'y': data_vio1.loc[data_vio1.TipoBeneficio == TipoBeneficio]['CantidadDeBeneficiarios'],
            'type': 'violin',
            'name': TipoBeneficio,
            'showlegend': True,
            } for i, TipoBeneficio in enumerate(list(set(groups.TipoBeneficio)))
           ],
)
fig.show()


# In[41]:


groups = data_vio1.groupby('Etnia').filter(lambda x: len(x) > 10).reset_index()

import plotly.graph_objects as go
#parametro hsl define el tipo de graficacion
#linespace tamaño de la pantalla
#shapes 
C = ['hsl('+str(h)+', 50%'+'50%)' for h in np.linspace(0,720,len(set(groups.CantidadDeBeneficiarios)))]
#creacion de variable tipo figura
fig = go.Figure(
    #contenedor de elementos graficos
    #tickangle parametros de inclinacion del texto
    layout = {'title': 'Cantidad de beneficiarios por Etnia',
              'xaxis': {'tickangle': -40},
              'yaxis': {'title': 'CantidadDeBeneficiarios'},
              'plot_bgcolor': 'rgb(230,230,230)',
              'shapes':[{'type': 'line',
                         'x0': -2,
                         'y0': np.nanmean(list(groups.CantidadDeBeneficiarios)),
                         'x1': 18,
                         'y1':np.nanmean(list(groups.CantidadDeBeneficiarios)),
                         'line':{'dash':'dashdot'}
                        }]
             },
    data = [{'y': data_vio1.loc[data_vio1.Etnia == Etnia]['CantidadDeBeneficiarios'],
            'type': 'violin',
            'name': Etnia,
            'showlegend': True,
            } for i, Etnia in enumerate(list(set(groups.Etnia)))
           ],
)
fig.show()


# In[42]:


groups = data_vio1.groupby('RangoEdad').filter(lambda x: len(x) > 10).reset_index()

import plotly.graph_objects as go
#parametro hsl define el tipo de graficacion
#linespace tamaño de la pantalla
#shapes 
C = ['hsl('+str(h)+', 50%'+'50%)' for h in np.linspace(0,720,len(set(groups.CantidadDeBeneficiarios)))]
#creacion de variable tipo figura
fig = go.Figure(
    #contenedor de elementos graficos
    #tickangle parametros de inclinacion del texto
    layout = {'title': 'Cantidad de beneficiarios por RangoEdad',
              'xaxis': {'tickangle': -40},
              'yaxis': {'title': 'CantidadDeBeneficiarios'},
              'plot_bgcolor': 'rgb(230,230,230)',
              'shapes':[{'type': 'line',
                         'x0': -2,
                         'y0': np.nanmean(list(groups.CantidadDeBeneficiarios)),
                         'x1': 18,
                         'y1':np.nanmean(list(groups.CantidadDeBeneficiarios)),
                         'line':{'dash':'dashdot'}
                        }]
             },
    data = [{'y': data_vio1.loc[data_vio1.RangoEdad == RangoEdad]['CantidadDeBeneficiarios'],
            'type': 'violin',
            'name': RangoEdad,
            'showlegend': True,
            } for i, RangoEdad in enumerate(list(set(groups.RangoEdad)))
           ],
)
fig.show()


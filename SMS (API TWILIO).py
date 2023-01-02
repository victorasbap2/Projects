#!/usr/bin/env python
# coding: utf-8

# # 💬 TESTE PARA ENVIO DE SMS COM API TWILIO 

# In[1]:


import os


# In[3]:


get_ipython().system('pip install twilio')


# In[4]:


from twilio.rest import Client


# In[6]:


account_sid = "AC1335462d196114ea7df5e2529ab2be07"
auth_token = "os.environ["TWILIO_AUTH_TOKEN"]"


# In[7]:


client = Client(account_sid, auth_token)


# In[8]:


client.messages.create(from_="+12056516179",
                       to= os.environ.get('CELL_PHONE_NUMBER'),
                       body='ATENCAO: MENSAGEM DE EVACUACAO - Com base no sistema SAFE SLOPES, notamos que a regiao em que sua residencia se localiza esta em uma area de risco de iminente deslizamento. Alertamos para que se retire da area imediatamente para evacuacao. Agradecemos sua compreensao!')


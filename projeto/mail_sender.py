import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email_config import smtp_host, smtp_port, username, password


def attach_file(msg, filepath):
    """Função para anexar um arquivo ao e-mail."""
    with open(filepath, 'rb') as attachment:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment', filename=filepath.split('/')[-1])
        msg.attach(part)


def mail_sender(msg_header,body_path,doc_path = None):

    # Criar mensagem
    msg = MIMEMultipart()
    msg['From'] = msg_header['From']
    msg['To'] = msg_header['To'] 
    msg['Subject'] = msg_header['Subject']

    # Corpo da mensagem
    with open(f'{body_path}', 'r', encoding='utf-8') as file:
        body = file.read()

    # Adicionar o corpo da mensagem ao e-mail
    msg.attach(MIMEText(body, 'html'))

    # Caminho para os arquivos PDF que serão anexados
    if doc_path != None:
        for doc in doc_path:
            attach_file(msg, doc)

    # Conectar ao servidor SMTP do Gmail e enviar o e-mail
    server = smtplib.SMTP(smtp_host, smtp_port)
    server.starttls()  # Iniciar conexão TLS
    server.login(username, password)
    server.sendmail(msg['From'], msg['To'], msg.as_string())
    server.quit()

if __name__ == '__main__':
    msg_header = {'From': username,'To': username, 'Subject': 'Test'}
    body_path = 'C:\\Users\\ntana\\Doutorado\\BMT\\projeto\\colacao\\colacao_body.html'
    doc_path = None
    mail_sender(msg_header,body_path,doc_path )
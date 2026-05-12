import os
from pathlib import Path

# Definir os caminhos para as pastas
pasta_labels = Path('labels/obj_train_data')
pasta_imagens = Path('images/ppfotos')

# Contadores para lhe dar um resumo no final
pares_apagados = 0

# Verificar se as pastas existem para evitar erros
if not pasta_labels.exists() or not pasta_imagens.exists():
    print("Erro: Uma das pastas (labels ou imagens) não foi encontrada. Verifique os caminhos.")
else:
    print("A iniciar a limpeza de ficheiros com 0KB...\n")
    
    # Percorrer todos os ficheiros .txt na pasta de labels
    for ficheiro_txt in list(pasta_labels.glob('*.txt')): 
        # (Usamos list() para evitar problemas ao apagar ficheiros enquanto iteramos sobre a pasta)
        
        # Verificar se o tamanho do ficheiro é 0 bytes
        if ficheiro_txt.stat().st_size == 0:
            
            # Construir o caminho para a imagem correspondente (.jpg)
            ficheiro_jpg = pasta_imagens / (ficheiro_txt.stem + '.jpg')
            
            # Se a imagem existir, apaga-a
            if ficheiro_jpg.exists():
                ficheiro_jpg.unlink()  # Apaga a imagem .jpg
                print(f"Apagada a imagem: {ficheiro_jpg}")
            else:
                print(f"Aviso: A imagem '{ficheiro_jpg.name}' não foi encontrada, mas a label será apagada na mesma.")
            
            # Apagar a label .txt com 0KB
            ficheiro_txt.unlink()
            print(f"Apagada a label:  {ficheiro_txt}\n")
            
            pares_apagados += 1

    print(f"Concluído! Foram limpos {pares_apagados} ficheiros de texto vazios (e as respetivas imagens).")
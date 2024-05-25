import argparse
import os
import subprocess
import torch
import tqdm
import transformers


def download_file(_id, dest, cache_dir):
    if os.path.exists(dest) or os.path.exists(os.path.join(cache_dir, dest)):
        print ("[Already exists] Skipping", dest)
        print ("If you want to download the file in another location, please specify a different path")
        return

    if os.path.exists(dest.replace(".zip", "")) or os.path.exists(os.path.join(cache_dir, dest.replace(".zip", ""))):
        print ("[Already exists] Skipping", dest)
        print ("If you want to download the file in another location, please specify a different path")
        return

    if "/" in dest:
        dest_dir = "/".join(dest.split("/")[:-1])
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)
    else:
        dest_dir = "."

    if _id.startswith("https://"):
        command = """wget -O %s %s""" % (dest, _id)
    else:
        command = """wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=%s' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id=%s" -O %s && rm -rf /tmp/cookies.txt""" % (_id, _id, dest)

    ret_code = subprocess.run([command], shell=True)
    if ret_code.returncode != 0:
        print("Download {} ... [Failed]".format(dest))
    else:
        print("Download {} ... [Success]".format(dest))

    if dest.endswith(".zip"):
        command = """unzip %s -d %s && rm %s""" % (dest, dest_dir, dest)

        ret_code = subprocess.run([command], shell=True)
        if ret_code.returncode != 0:
            print("Unzip {} ... [Failed]".format(dest))
        else:
            print("Unzip {} ... [Success]".format(dest))



def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        type=str,
                        default=".cache/factscore")
    parser.add_argument('--model_dir',
                        type=str,
                        default=".cache/factscore")

    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    download_file("1IseEAflk1qqV0z64eM60Fs3dTgnbgiyt", "demos.zip", args.data_dir)
    download_file("1enz1PxwxeMr4FRF9dtpCPXaZQCBejuVF", "data.zip", args.data_dir)
    download_file("1mekls6OGOKLmt7gYtHs0WGf5oTamTNat", "enwiki-20230401.db", args.data_dir)

  

    # download the roberta_stopwords.txt file
    subprocess.run(["wget https://raw.githubusercontent.com/shmsw25/FActScore/main/roberta_stopwords.txt"], shell=True)

    # move the files to the data directory
    subprocess.run(["mv demos %s" % args.data_dir], shell=True)
    subprocess.run(["mv enwiki-20230401.db %s" % args.data_dir], shell=True)


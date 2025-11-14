import numpy as np
from sound import SoundClass
from  plot_bar import *
from linear_model import LinearModel
from linear_filter_bank import FilterBank

base_path = "results/linear"
normalizer_name = f"{base_path}/norm_stats.npz"
output_name = f"{base_path}/correct_predictions.png"
max_frames = 83


def tester():
    linear_model = LinearModel()
    filter_bank = FilterBank()
    data, labels = filter_bank.get_training_features()

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0) + 1e-8
    np.savez(normalizer_name, mean=mean, std=std)
    data = (data - mean) / std
    labels = linear_model.labelEncoder(labels)
    linear_model.train_model(data, labels)

def test_model(r_in, fs_in, model, LE):
    linear_model = LinearModel()

    filter_bank = FilterBank()
    test_data = filter_bank.feature_extraction(r_in, fs_in, False)




    # print("test_data shape before norm:", np.array(test_data).shape)
    #


    stats = np.load(normalizer_name)
    mean, std = stats['mean'], stats['std']
    # print(stats['mean'][:5], stats['std'][:5])
    # print("mean/std shape:", mean.shape, std.shape)


    test_data = np.array(test_data).reshape(-1, 64)
    test_data = (test_data - mean) / std

    pred = model.predict(test_data, verbose=0)
    top_five = linear_model.top5(pred, LE)
    for i in top_five:
        print(f"Pred at {i}")

    predicted_id = np.argmax(pred, axis=1)
    # print("predicted class:", predicted_id)
    predicted_name = LE.inverse_transform(predicted_id)[0]

    confidence = f"{np.max(pred) * 100:.2f}%"
    # print(f"Predicted Name: {predicted_name}")
    # print(f"Confidence: {confidence}")
    print("=====")
    return predicted_name, confidence



if __name__ == "__main__":
    tester()
    # r_in, fs_in = SoundClass().record_ns(fs=44100,seconds= 3)
    # test_model(r_in, fs_in, model, LE)

    # results = []
    # for audio_file in sorted(glob.glob('test_audios/*.wav')):
    #     r_in, fs_in = sf.read(audio_file, dtype='float32')
    #     if r_in.ndim == 2:
    #         r_in = r_in.mean(axis=1)
    #     # for i in range(5):
    #     print(f"Testing with audio, {audio_file} and sample rate {fs_in}")
    #
    #     predicted_name, confidence = test_model(r_in, fs_in, model, LE)
    #     audio_file = audio_file.split('/')[-1]
    #     actual_name = ''.join([c for c in audio_file.split('.')[0] if not c.isdigit()])
    #     results.append({
    #             'Actual Name': actual_name,
    #             'Predicted Name': predicted_name.strip(),
    #         })
    #
    #
    #
    # pprint.pprint(results)


    # results = []
    # for audio_file in sorted(glob.glob('names_audio_wav/*.wav')):
    #     print(f"\n🎤 File: {audio_file}")
    #     r_in, fs_in = sf.read(audio_file, dtype='float32')
    #     predicted_name, confidence = test_model(r_in, fs_in, model, LE)
    #
    #     file_name = audio_file.split('/')[-1]
    #     actual_name = ''.join([c for c in file_name.split('.')[0] if not c.isdigit()])
    #
    #     results.append({
    #         'Actual Name': actual_name,
    #         'Predicted Name': predicted_name
    #     })
    # plot_bar_chart(output_name, results)
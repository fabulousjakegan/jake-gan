import srt
from pydub import AudioSegment


# load a csv into a list
def find_sentences(filename):
    time_segments = list()
    with open(filename, 'r') as f:
        transcript = srt.parse(f)
        transcript = list(transcript)
    end = False
    for i, segment in enumerate(transcript):
        if '.' in segment.content:
            try:
                time_segments.append(transcript[i+1].start)
            except:
                time_segments.append(segment.end)

    return time_segments


def split_wav_file(filename, time_segments):
    t1 = 0
    src = filename[:-4]
    newAudio = AudioSegment.from_wav(filename)
    for i, end in enumerate(time_segments):
        t2 = (end.seconds * 1000) + (end.microseconds / 1000)
        segmentAudio = newAudio[t1:t2]
        segmentAudio.export("jake_" + src + "_" + str(i) + ".wav", format="wav")
        t1 = t2


split_wav_file("lecture3.wav", find_sentences("lecture3.srt"))
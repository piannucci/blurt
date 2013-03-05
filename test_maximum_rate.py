import time
rate = 0
length = 1500
input_octets = np.random.random_integers(0,255,length)
output = wifi.encode(input_octets, rate)
N = output.size
trials = 10
t0 = time.time(); [(None, wifi.encode(input_octets, rate))[0] for i in xrange(trials)]; t1 = time.time()
samples_encoded = trials * N
time_elapsed_encode = t1 - t0
max_sample_rate_encode = samples_encoded / time_elapsed_encode
t0 = time.time(); [(None, wifi.decode(output))[0] for i in xrange(trials)]; t1 = time.time()
samples_decoded = trials * N
time_elapsed_decode = t1 - t0
max_sample_rate_decode = samples_decoded / time_elapsed_decode
print max_sample_rate_encode, max_sample_rate_decode
import cProfile as profile
profile.run('wifi.decode(output)')

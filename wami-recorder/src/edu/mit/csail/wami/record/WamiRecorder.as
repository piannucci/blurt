/* 
* Copyright (c) 2011
* Spoken Language Systems Group
* MIT Computer Science and Artificial Intelligence Laboratory
* Massachusetts Institute of Technology
*
* Permission is hereby granted, free of charge, to any person
* obtaining a copy of this software and associated documentation
* files (the "Software"), to deal in the Software without
* restriction, including without limitation the rights to use, copy,
* modify, merge, publish, distribute, sublicense, and/or sell copies
* of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
* NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
* BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
* ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
* CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/
package edu.mit.csail.wami.record
{
	import edu.mit.csail.wami.audio.AudioFormat;
	import edu.mit.csail.wami.audio.EncodePipe;
	import edu.mit.csail.wami.audio.IAudioContainer;
	import edu.mit.csail.wami.audio.WaveContainer;
	import edu.mit.csail.wami.client.WamiParams;
	import edu.mit.csail.wami.utils.BytePipe;
	import edu.mit.csail.wami.utils.External;
	import edu.mit.csail.wami.utils.Pipe;
	import edu.mit.csail.wami.utils.StateListener;
	
	import flash.events.SampleDataEvent;
	import flash.events.StatusEvent;
	import flash.media.Microphone;
	import flash.media.SoundCodec;
	import flash.utils.Endian;
	import flash.utils.clearInterval;
	import flash.utils.setInterval;
	
	public class WamiRecorder implements IRecorder
	{
		private static var CHUNK_DURATION_MILLIS:Number = 200;
		
		private var mic:Microphone = null;
		private var params:WamiParams;
		private var audioPipe:Pipe;

		// For adding some audio padding to start and stop.
		private var stopInterval:uint;
		private var listening:Boolean = false;
		
		// To determine if the amount of audio recorded matches up with 
		// the length of time we've recorded (i.e. not dropping any frames)
		private var handled:uint;
		private var startTime:Date;
		private var stopTime:Date;
		private var listener:StateListener;

		public function WamiRecorder(mic:Microphone, params:WamiParams)
		{	
			this.params = params;
			this.mic = mic;
			mic.addEventListener(StatusEvent.STATUS, onMicStatus);
			
			if (getChunkSize() <= 0)
			{
				throw new Error("Desired duration is too small, even for streaming chunks: " + getChunkSize());
			}
		}
		
		/**
		 * The WAMI recorder can listen constantly, keeping a buffer of the last
		 * few milliseconds of audio.
		 */
		public function listen():void {
			if (!listening) {
				mic.rate = AudioFormat.toRoundedRate(params.format.rate);
				mic.codec = SoundCodec.NELLYMOSER;  // Just to clarify 5, 8, 11, 16, 22 and 44 kHz
				mic.setSilenceLevel(0, 10000);
				mic.addEventListener(SampleDataEvent.SAMPLE_DATA, sampleHandler);
				External.debug("Listening...");
				listening = true;
			}
		}
		
		public function unlisten():void {
			if (listening) {
				mic.removeEventListener(SampleDataEvent.SAMPLE_DATA, sampleHandler);
				listening = false;
				External.debug("Unlistening.");
			}
		}
		
		protected function onMicStatus(event:StatusEvent):void
		{
			External.debug("status: " + event.code);
			if (event.code == "Microphone.Unmuted") 
			{
				listen();
			} else if (event.code == "Microphone.Muted") {
				unlisten();
			}
		}
		
		public function start(listener:StateListener):void 
		{	
			// Forces security if mic is still muted in debugging mode.
			listen();

			// Flash might be able to decide on a different sample rate
			// than the one you suggest depending on your audio card...
			params.format.rate = AudioFormat.fromRoundedRate(mic.rate);
			External.debug("Recording at rate: " + params.format.rate);

			stop(true);
			audioPipe = createAudioPipe(listener);

			listener.started();

			handled = 0;
			startTime = new Date();
		}
		
		public function createAudioPipe(listener:StateListener):Pipe
		{
			this.listener = listener;
			
			var post:Pipe;
			var container:IAudioContainer;
            post = new SingleExternal(listener);
            container = new WaveContainer();
			
			// Setup the audio pipes.  A transcoding pipe converts floats
			// to shorts and passes them on to a chunking pipe, which spits
			// out chunks to a pipe that possibly adds a WAVE header
			// before passing the chunks on to a pipe that does HTTP posts.
			var pipe:Pipe = new EncodePipe(params.format, container);
			pipe.setSink(new ChunkPipe(getChunkSize()))
				.setSink(post);

			return pipe;
		}
		
		internal function sampleHandler(evt:SampleDataEvent):void
		{
			evt.data.position = 0;
			try 
			{
				if (audioPipe)
				{
					audioPipe.write(evt.data);
					handled += evt.data.length / 4;
				}
			}
			catch (error:Error)
			{
				audioPipe = null;
				stop(true);
				listener.failed(error);
			}
		}
		
		public function stop(force:Boolean = false):void 
		{
			clearInterval(stopInterval);

			if (force) 
			{
				reallyStop();
			} 
			else 
			{
				stopInterval = setInterval(function():void {
					clearInterval(stopInterval);
					reallyStop();
				}, 0);
			}
		}
		
		public function level():int 
		{
			if (!audioPipe) return 0;
			return mic.activityLevel;
		}
		
		private function reallyStop():void
		{
			if (!audioPipe) return;

			try {
				audioPipe.close();
			} catch(error:Error) {
				listener.failed(error);
			}
			
			audioPipe = null;
			validateAudioLength();

            unlisten();
		}
		
		private function validateAudioLength():void
		{
			stopTime = new Date();
			var seconds:Number = ((stopTime.time - startTime.time) / 1000.0);
			var expectedSamples:uint = uint(seconds*params.format.rate);
			External.debug("Expected Samples: " + expectedSamples + " Actual Samples: " + handled);
			startTime = null;
			stopTime = null;
		}
		
		private function getBytesPerSecond():uint 
		{
			return params.format.channels * (params.format.bits/8) * params.format.rate;
		}
		
		private function getChunkSize():uint
		{
			return int.MAX_VALUE;
		}
		
	}
}

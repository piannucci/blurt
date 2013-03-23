/*
* Base64 - 1.1.0
*
* Copyright (c) 2006 Steve Webster
*
* Permission is hereby granted, free of charge, to any person obtaining a copy of
* this software and associated documentation files (the "Software"), to deal in
* the Software without restriction, including without limitation the rights to
* use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
* the Software, and to permit persons to whom the Software is furnished to do so,
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
* FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
* COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
* IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
* CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
package edu.mit.csail.wami.utils {
	import flash.utils.ByteArray;

	public class Base64 {
		private static const BASE64_CHARS:String = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=";

		public static function encodeByteArray(data:ByteArray):String {
			data.position = 0;
            var n:uint = data.bytesAvailable/3;
            var outputBuffer:ByteArray = new ByteArray();
            var d:uint = 0;
            var i:uint;
            for (i = 0; i < n; i++) {
                d  = data.readUnsignedByte() << 16;
                d |= data.readUnsignedByte() << 8;
                d |= data.readUnsignedByte();
                outputBuffer.writeByte(BASE64_CHARS.charCodeAt((d & 0xfc0000) >> 18));
                outputBuffer.writeByte(BASE64_CHARS.charCodeAt((d & 0x03f000) >> 12));
                outputBuffer.writeByte(BASE64_CHARS.charCodeAt((d & 0x000fc0) >>  6));
                outputBuffer.writeByte(BASE64_CHARS.charCodeAt((d & 0x00003f) >>  0));
            }
            d = 0;
            i = 0;
			while (data.bytesAvailable > 0) {
                d |= data.readUnsignedByte() << ((2-i) << 3);
                i++;
            }
            switch (i) {
                case 0:
                    break;
                case 1:
                    outputBuffer.writeByte(BASE64_CHARS.charCodeAt((d & 0xfc0000) >> 18));
                    outputBuffer.writeByte(BASE64_CHARS.charCodeAt((d & 0x030000) >> 12));
                    outputBuffer.writeByte(BASE64_CHARS.charCodeAt(64));
                    outputBuffer.writeByte(BASE64_CHARS.charCodeAt(64));
                    break;
                case 2:
                    outputBuffer.writeByte(BASE64_CHARS.charCodeAt((d & 0xfc0000) >> 18));
                    outputBuffer.writeByte(BASE64_CHARS.charCodeAt((d & 0x03f000) >> 12));
                    outputBuffer.writeByte(BASE64_CHARS.charCodeAt((d & 0x000f00) >>  6));
                    outputBuffer.writeByte(BASE64_CHARS.charCodeAt(64));
            }
            return outputBuffer.toString();

			//var output:String = "";
			//var dataBuffer:Array = new Array(3);
			//var outputBuffer:Array = new Array(4);
			//while (data.bytesAvailable > 0) {
            //    var k:uint = (data.bytesAvailable >= 3) ? 3 : data.bytesAvailable;
			//	for (var i:uint = 0; i < k; i++)
			//		dataBuffer[i] = data.readUnsignedByte();
			//	outputBuffer[0] = (dataBuffer[0] & 0xfc) >> 2;
			//	outputBuffer[1] = ((dataBuffer[0] & 0x03) << 4) | ((dataBuffer[1]) >> 4);
			//	outputBuffer[2] = ((dataBuffer[1] & 0x0f) << 2) | ((dataBuffer[2]) >> 6);
			//	outputBuffer[3] = dataBuffer[2] & 0x3f;
			//	for (var j:uint = k; j < 3; j++)
			//		outputBuffer[j + 1] = 64;
			//	for (var k:uint = 0; k < outputBuffer.length; k++)
			//		output += BASE64_CHARS.charAt(outputBuffer[k]);
			//}
			//return output;
		}

		public function Base64() {
			throw new Error("Base64 class is static container only");
		}
	}
}

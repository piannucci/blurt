<?php
if (isset($_FILES['myFile'])) {
    // Example:
    $addr = $_SERVER['REMOTE_ADDR'];
    $addr = str_replace(':', '_', $addr);
    $addr = str_replace('.', '-', $addr);
    $fn = 'uploads/' . $addr . '.wav';
    rename($_FILES['myFile']['tmp_name'], $fn);
    chmod($fn, 0644);
    print shell_exec("/usr/bin/blurt " . getcwd() . '/' . $fn);
    exit;
}
?><!DOCTYPE html>
<html>
<head>
    <title>Audio Sample</title>
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
    <script type="text/javascript" src="recorder.js"> </script>
    <script type="text/javascript">
        window.onload = function() {
            function sendFile(file) {
                var uri = "/index.php";
                var xhr = new XMLHttpRequest();
                var fd = new FormData();
                 
                xhr.open("POST", uri, true);
                xhr.onreadystatechange = function() {
                    if (xhr.readyState == 4 && xhr.status == 200) {
                        status.innerText = xhr.responseText;
                    }
                };
                fd.append('myFile', file);
                // Initiate a multipart/form-data upload
                xhr.send(fd);
            }

			var onFail = function(e) {
                status.innerText = 'Rejected: ' + String(e);
			};

			var onSuccess = function(s) {
				var context = new webkitAudioContext();
				var mediaStreamSource = context.createMediaStreamSource(s);
				recorder = new Recorder(mediaStreamSource);
				recorder.record();
                status.innerText = 'Recording...';
                window.setTimeout(stopRecording, 5000);
			}

			window.URL = window.URL || window.webkitURL;
			navigator.getUserMedia  = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;

			var recorder;
            var status = document.querySelector('#status');
            var args = {audio: true};

			function startRecording() {
				if (navigator.getUserMedia) {
					navigator.getUserMedia(args, onSuccess, onFail);
				} else {
					console.log('navigator.getUserMedia not present');
				}
			}

			function stopRecording() {
                recorder.stop();
				recorder.exportWAV(function(s) {
                    sendFile(s);
                });
                status.innerText = 'Uploading...';
			}

            startRecording();
        }
    </script>
</head>
<body>
    <div>
        <div id="status">Requesting permission to record...</div>
    </div>
</body>
</html>

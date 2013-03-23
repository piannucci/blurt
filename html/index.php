<?php
if (isset($_FILES['myFile'])) {
    // Example:
    $addr = $_SERVER['REMOTE_ADDR'];
    $addr = str_replace(':', '_', $addr);
    $addr = str_replace('.', '-', $addr);
    $fn = 'uploads/' . $addr . '.wav';
    rename($_FILES['myFile']['tmp_name'], $fn);
    chmod($fn, 0644);
    $f = fsockopen("unix:///blurt/socket");
    fwrite($f, realpath($fn));
    $result = '';
    $length = intval(fread($f, 8));
    while (strlen($result) < $length)
    {
        $result .= fread($f, min(1024, $length - strlen($result)));
    }
    print $result;
    fclose($f);
    exit;
}
?>
<html>
    <head>
        <title>Sign up</title>
        <link rel="stylesheet" type="text/css" href="ui.css">
        <script type="text/javascript" src="recorder.js"> </script>
        <script type="text/javascript" src="https://ajax.googleapis.com/ajax/libs/swfobject/2.2/swfobject.js"></script>
        <script type="text/javascript" src="wamirecorder.js"> </script>
        <script type="text/javascript">
        <!--
        var activeScreen = 'intro';
        function makeActive(s) {
            oldActive = document.querySelector('#'+activeScreen);
            oldActive.className = oldActive.className.replace(/(?:^|\s)activeScreen(?!\S)/g, '');
            newActive = document.querySelector('#'+s);
            newActive.className += ' activeScreen';
            oldHeight = oldActive.scrollHeight;
            newHeight = newActive.scrollHeight;
            form = document.querySelector('#form');
            form.style.height = newHeight;
            activeScreen = s;
            document.querySelector('#middle').style.backgroundImage = null;
        }

        window.URL = window.URL || window.webkitURL;
        navigator.getUserMedia  = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;
        var recorder = null;
        var wamiInitialized = false;
        var awaitingRecording = true;
        var recordingTimeout = null;
        var inProgressXHR = null;
        function status(s) {
            var el = document.querySelector('#status');
            if (el.innerText != undefined)
                el.innerText = s;
            else
                el.textContent = s;
        }

        function completePost(responseText) {
            console.log(responseText);
            imageLength = parseInt(responseText.substr(0, 8), 10);
            image = responseText.substr(8, imageLength);
            textLength = parseInt(responseText.substr(8+imageLength, 8), 10);
            text = responseText.substr(16+imageLength, textLength);
            status(text);
            document.querySelector('#middle').style.backgroundImage = 'url(\'' + image + '\')';
            document.querySelector('#form').style.height = document.querySelector('#record').scrollHeight;
            inProgressXHR = null;
        }

        function beginPost(file) {
            abortPost();
            var uri = "/index.php";
            var xhr = new XMLHttpRequest();
            var fd = new FormData();

            xhr.open("POST", uri, true);
            xhr.onreadystatechange = function() {
                if (xhr.readyState == 4 && xhr.status == 200)
                    completePost(xhr.responseText);
            };
            inProgressXHR = xhr;
            fd.append('myFile', file);
            xhr.send(fd);
        }

        function abortPost() {
            if (inProgressXHR != null)
                inProgressXHR.abort();
            inProgressXHR = null;
        }

        function startAnimation(s) {
            s = 'notefganim' + s;
            notefg = document.querySelector('#notefg');
            notefg.className = notefg.className.replace(/(?:^|\s)notefg\S*(?!\S)/g, '');
            notefg.className += ' ' + s;
            notefg.parentNode.replaceChild(notefg.cloneNode(true), notefg);
        }

        function wamiStarted() {
        }

        function wamiFinished(result) {
            result = result[0];
            if (awaitingRecording) {
                result = atob(result);
                var buffer = new ArrayBuffer(result.length);
                var intarray = new Uint8Array(buffer);
                for(var i=0; i<result.length; i++)
                    intarray[i] = result.charCodeAt(i);
                beginPost(new Blob([buffer], { type: 'audio/wav' }));
            } else {
                console.log('Got recording from Wami after cancel');
            }
        }

        function record() {
            clearRecordingTimeout();
            if (recorder != null) {
                recorder.clear();
                recorder.record();
            } else if (wamiInitialized) {
                Wami.startRecording(startedfn="wamiStarted", finishedfn="wamiFinished", failedfn="wamiFailed");
            } else {
                return;
            }
            startAnimation('record');
            status('Recording...');
            recordingTimeout = window.setTimeout(stopRecording, 5000);
            awaitingRecording = true;
        }

        function getUserMediaSuccess(s) {
            var context = new (window.webkitAudioContext || window.mozAudioContext)();
            if (context.createMediaStreamSource) {
                var mediaStreamSource = context.createMediaStreamSource(s);
                recorder = new Recorder(mediaStreamSource);
                wamiInitialized = false;
                record();
            } else {
                tryWamiSetup();
            }
        }

        function wamiSuccess() {
            recorder = null;
            wamiInitialized = true;
            record();
        }

        function getUserMediaFail(e) {
            tryWamiSetup();
        };

        function tryWamiSetup() {
            try {
                Wami.setup({id:"wami", onLoaded: wamiSuccess});
            } catch (err) {
                status('Browser not supported');
                startAnimation('fail');
                return false;
            }
            return true;
        }

        function startRecording() {
            if (recorder == null && !wamiInitialized) {
                if (navigator.getUserMedia) {
                    var args = {audio: true};
                    navigator.getUserMedia(args, getUserMediaSuccess, getUserMediaFail);
                } else {
                    tryWamiSetup();
                }
            } else {
                record();
            }
            makeActive('record');
        }

        function clearRecordingTimeout() {
            if (recordingTimeout != null)
                window.clearTimeout(recordingTimeout);
            recordingTimeout = null;
        }

        function stopRecording() {
            clearRecordingTimeout();
            if (recorder != null) {
                recorder.stop();
                recorder.exportWAV(beginPost);
                recorder.clear();
            } else if (wamiInitialized) {
                Wami.stopRecording();
            }
            window.setTimeout(function () {awaitingRecording = false;}, 1000);
            status('Uploading...');
        }

        function cancelRecord() {
            makeActive('intro')
            awaitingRecording = false;
            clearRecordingTimeout();
            if (recorder != null) {
                recorder.stop();
                recorder.clear();
            } else if (wamiInitialized) {
                Wami.stopRecording();
            }
            abortPost();
        }

        window.onload = function() {
            setTimeout(makeActive, 100, 'intro');

            note = document.querySelector('.note');
            notefg = document.querySelector('#notefg');
            noteshadow = document.querySelector('.noteshadow');
            if (note.style.webkitBackgroundClip == null) {
                note.style.color = 'transparent';
                notefg.style.color = 'transparent';
                note.style.clipPath = 'url(#c1)';
                notefg.style.clipPath = 'url(#c1)';
            }
        }
        //-->
        </script>
    </head>
    <body>
        <div id="outer">
            <div id="middle">
                <form class="form-container" id="form" action="javascript:return false;">
                    <div class="screens">
                        <div class="form-screen activeScreen rootScreen" id="intro">
                            <div class="form-title"><h2>Welcome to Initrode</h2></div>
                            <div class="form-title" style="margin-top: 15px;">What brings you to our network?</div>
                            <div class="submit-container">
                                <input class="submit-button" type="button" value="I have an account" onclick="makeActive('signin');" style="margin-bottom:5px;"/><br/>
                                <input class="submit-button" type="button" value="I'm in the building" onclick="startRecording();" style="margin-bottom:5px;"/><br/>
                                <input class="submit-button" type="button" value="I'm feeling lucky" onclick="makeActive('lucky');"/>
                            </div>
                        </div>
                        <div class="form-screen subScreen" id="signin">
                            <div class="form-title"><h2>Sign in</h2></div>
                            <div class="form-title">Name</div>
                            <input class="form-field" type="text" name="firstname" /><br />
                            <div class="form-title">Email</div>
                            <input class="form-field" type="text" name="email" /><br />
                            <div class="submit-container">
                                <input class="submit-button" type="button" value="Cancel" onclick="makeActive('intro');" style="width:140px;margin-right:10px;"/>
                                <input class="submit-button" type="button" value="Sign in" onclick="makeActive('intro');" style="width:140px;"/>
                            </div>
                        </div>
                        <div class="form-screen subScreen" id="record">
                            <div class="form-title"><h2>Acoustic Authorization</h2></div>
                            <div class="noteshadow notefont">&#x266C;</div>
                            <div class="note notefont">&#x266C;</div>
                            <div id="notefg" class="notefont">&#x266C;</div>
                            <div id="wami"></div>
                            <div class="form-title" id="status">Requesting permission to record</div>
                            <input class="submit-button" type="button" value="Cancel" onclick="cancelRecord();"/>
                        </div>
                        <div class="form-screen subScreen" id="lucky">
                            <div class="form-title"><h2>Limited Access</h2></div>
                            <div class="form-title" style="width:300px;">You may return to this page at any time.</div>
                            <div class="submit-container">
                                <input class="submit-button" type="button" value="Cancel" onclick="makeActive('intro');" style="width:140px;margin-right:10px;"/>
                                <input class="submit-button" type="button" value="Sign in" onclick="makeActive('intro');" style="width:140px;"/>
                            </div>
                        </div>
                    </div>
                </form>
            </div>
        </div>
        <svg height="0">
          <clipPath id="c1" clipPathUnits="userSpaceOnUse">
            <text x="0" y="160" class="notefont">&#x266C;</text>
          </clipPath>
        </svg>
    </body>
</html>

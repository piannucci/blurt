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
    //print shell_exec("/usr/bin/blurt " . getcwd() . '/' . $fn);
    exit;
}
?>
<html>
    <head>
        <title>Sign up</title>
        <style type="text/css">
            body {
                background: #87e0fd; /* Old browsers */
                background: url('noise.png') center center, -moz-radial-gradient(center, ellipse cover,  #5A6978 0%, #293948 100%); /* FF3.6+ */
                background: url('noise.png') center center, -webkit-gradient(radial, center center, 0px, center center, 100%, color-stop(0%,#5A6978), color-stop(100%,#293948)); /* Chrome,Safari4+ */
                background: url('noise.png') center center, -webkit-radial-gradient(center, ellipse cover,  #5A6978 0%,#293948 100%); /* Chrome10+,Safari5.1+ */
                background: url('noise.png') center center, -o-radial-gradient(center, ellipse cover,  #5A6978 0%,#293948 100%); /* Opera 12+ */
                background: url('noise.png') center center, -ms-radial-gradient(center, ellipse cover,  #5A6978 0%,#293948 100%); /* IE10+ */
                background: url('noise.png') center center, radial-gradient(ellipse at center,  #5A6978 0%,#293948 100%); /* W3C */
            }
            body, html {
                height: 100%;
                margin: 0;
                padding: 0;
            }
            #outer {
                display: table;
                height: 100%;
                width: 100%;
                overflow: hidden;
                position: relative;
            }
            #outer[id] {
                display: table;
                position: static;
            }
            #middle {
                position: absolute;
                top: 50%;
                background-position: center left;
                background-repeat: repeat-x;
                image-rendering: -webkit-optimize-contrast;
                image-rendering: optimize-contrast;
            }
            #middle[id] {
                display: table-cell;
                vertical-align: middle;
                width: 100%;
                position: static;
                overflow: auto;
            }
            .form-container {
                overflow: hidden;
                position: relative;
                top: -50%;
                margin-left: auto; margin-right: auto;
                border: 1px solid #f2e3d2;
                background: #c9b7a2;
                background: url('noise2.png'), -webkit-gradient(linear, left top, left bottom, from(#f2e3d2), to(#c9b7a2));
                background: url('noise2.png'), -webkit-linear-gradient(top, #f2e3d2, #c9b7a2);
                background: url('noise2.png'), -moz-linear-gradient(top, #f2e3d2, #c9b7a2);
                background: url('noise2.png'), -ms-linear-gradient(top, #f2e3d2, #c9b7a2);
                background: url('noise2.png'), -o-linear-gradient(top, #f2e3d2, #c9b7a2);
                background-image: -ms-linear-gradient(top, #f2e3d2 0%, #c9b7a2 100%);
                -webkit-border-radius: 8px;
                -moz-border-radius: 8px;
                border-radius: 8px;
                -webkit-box-shadow: rgba(000,000,000,0.9) 0 1px 2px, inset rgba(255,255,255,0.4) 0 0px 0;
                -moz-box-shadow: rgba(000,000,000,0.9) 0 1px 2px, inset rgba(255,255,255,0.4) 0 0px 0;
                box-shadow: rgba(000,000,000,0.9) 0 1px 2px, inset rgba(255,255,255,0.4) 0 0px 0;
                font-family: 'Helvetica Neue',Helvetica,sans-serif;
                text-decoration: none;
                vertical-align: middle;
                min-width:300px;
                padding:20px;
                width:300px;
                display: table;
                height: 20px;
                -webkit-transition: all .5s ease-in-out;
                -moz-transition: all .5s ease-in-out;
            }
            .form-field {
                border: 1px solid #c9b7a2;
                background: #e4d5c3;
                -webkit-border-radius: 4px;
                -moz-border-radius: 4px;
                border-radius: 4px;
                color: #c9b7a2;
                -webkit-box-shadow: rgba(255,255,255,0.4) 0 1px 0, inset rgba(000,000,000,0.7) 0 0px 0px;
                -moz-box-shadow: rgba(255,255,255,0.4) 0 1px 0, inset rgba(000,000,000,0.7) 0 0px 0px;
                box-shadow: rgba(255,255,255,0.4) 0 1px 0, inset rgba(000,000,000,0.7) 0 0px 0px;
                padding:8px;
                margin-bottom:20px;
                width:298px;
            }
            .form-field:focus {
                background: #fff;
                color: #725129;
            }
            .form-container h2 {
                text-shadow: #fdf2e4 0 1px 0;
                font-size:18px;
                margin: 0 0 10px 0;
                font-weight:bold;
                text-align:center;
            }
            .form-title {
                margin-bottom:10px;
                color: #725129;
                text-shadow: #fdf2e4 0 1px 0;
            }
            .submit-container {
                margin:0px 0;
                text-align:center;
            }
            .submit-button {
                border: 1px solid #447314;
                background: #6aa436;
                background: url('noise.png'), -webkit-gradient(linear, left top, left bottom, from(#8dc059), to(#6aa436));
                background: url('noise.png'), -webkit-linear-gradient(top, #8dc059, #6aa436);
                background: url('noise.png'), -moz-linear-gradient(top, #8dc059, #6aa436);
                background: url('noise.png'), -ms-linear-gradient(top, #8dc059, #6aa436);
                background: url('noise.png'), -o-linear-gradient(top, #8dc059, #6aa436);
                background-image: -ms-linear-gradient(top, #8dc059 0%, #6aa436 100%);
                -webkit-border-radius: 4px;
                -moz-border-radius: 4px;
                border-radius: 4px;
                -webkit-box-shadow: rgba(255,255,255,0.4) 0 1px 0, inset rgba(255,255,255,0.4) 0 1px 0;
                -moz-box-shadow: rgba(255,255,255,0.4) 0 1px 0, inset rgba(255,255,255,0.4) 0 1px 0;
                box-shadow: rgba(255,255,255,0.4) 0 1px 0, inset rgba(255,255,255,0.4) 0 1px 0;
                text-shadow: #addc7e 0 1px 0;
                color: #31540c;
                font-family: helvetica, serif;
                padding: 8.5px 18px;
                font-size: 16px;
                text-decoration: none;
                vertical-align: middle;
                width:300px;
            }
            .submit-button:hover {
                border: 1px solid #447314;
                text-shadow: #31540c 0 1px 0;
                background: #6aa436;
                background: url('noise.png'), -webkit-gradient(linear, left top, left bottom, from(#8dc059), to(#6aa436));
                background: url('noise.png'), -webkit-linear-gradient(top, #8dc059, #6aa436);
                background: url('noise.png'), -moz-linear-gradient(top, #8dc059, #6aa436);
                background: url('noise.png'), -ms-linear-gradient(top, #8dc059, #6aa436);
                background: url('noise.png'), -o-linear-gradient(top, #8dc059, #6aa436);
                background-image: -ms-linear-gradient(top, #8dc059 0%, #6aa436 100%);
                color: #fff;
            }
            .submit-button:active {
                text-shadow: #31540c 0 1px 0;
                border: 1px solid #447314;
                background: #8dc059;
                background: url('noise.png'), -webkit-gradient(linear, left top, left bottom, from(#6aa436), to(#6aa436));
                background: url('noise.png'), -webkit-linear-gradient(top, #6aa436, #8dc059);
                background: url('noise.png'), -moz-linear-gradient(top, #6aa436, #8dc059);
                background: url('noise.png'), -ms-linear-gradient(top, #6aa436, #8dc059);
                background: url('noise.png'), -o-linear-gradient(top, #6aa436, #8dc059);
                background-image: -ms-linear-gradient(top, #6aa436 0%, #8dc059 100%);
                color: #fff;
            }
            .screens {
                position: relative;
            }
            .form-screen {
                position:absolute;
                -webkit-transition: all .5s ease-in-out;
                -moz-transition: all .5s ease-in-out;
            }
            .activeScreen {
                -webkit-transform: translate(0px, 0);
                -moz-transform: translate(0px, 0);
            }
            .rootScreen:not(.activeScreen) {
                -webkit-transform: translate(-340px, 0);
                -moz-transform: translate(-340px, 0);
            }
            .subScreen:not(.activeScreen) {
                -webkit-transform: translate(340px, 0);
                -moz-transform: translate(340px, 0);
            }
            .noteshadow {
                position:absolute;
                left: 80px;
                top: 12px;
                font-size:140px;
                text-shadow: #eed 0 2px 2px, #221 0 -1px 1px;
            }
            .note {
                position:absolute;
                left: 80px;
                top: 12px;
                font-size:140px;
                background: -webkit-linear-gradient(#725129, #392814);
                background: -moz-linear-gradient(#725129, #392814);
                -webkit-background-clip: text;
                -moz-background-clip: text;
                -webkit-text-fill-color: transparent;
                -moz-text-fill-color: transparent;
                text-shadow: none;
            }
            @-webkit-keyframes successanim {
                from { background-position: -120px 175px; }
                10%  { background-position: -100px 160px; }
                20%  { background-position:  -80px 145px; }
                30%  { background-position: -100px 130px; }
                40%  { background-position: -120px 115px; }
                50%  { background-position: -100px 100px; }
                60%  { background-position:  -80px  85px; }
                70%  { background-position: -100px  70px; }
                80%  { background-position: -120px  55px; }
                90%  { background-position: -100px  40px; }
                to   { background-position:  -80px  25px; }
            }
            #notefg {
                position:absolute;
                left: 80px;
                top: 12px;
                font-size:140px;
                -webkit-background-clip: text;
                -moz-background-clip: text;
                -webkit-text-fill-color: transparent;
                -moz-text-fill-color: transparent;
            }
            .notefganim {
                -webkit-animation-name: successanim;
                -webkit-animation-timing-function: linear;
                -webkit-animation-duration: 5s;
                -webkit-animation-iteration-count: 1;
                -webkit-animation-fill-mode: forwards;
                background: url('drawing.svg');
                background-repeat: no-repeat;
            }
            .notefgfailanim {
                background: -webkit-linear-gradient(#ff0000, #800000);
                background-position:  0px 0px;
            }
            #status {
                text-align: center;
                margin-top:180px;
                width: 300px;
                max-width: 300px;
                min-height: 20px;
            }
            .notemask {
                clip-path: url(#c1);
                color: transparent !important;
            }
        </style>
        <script type="text/javascript" src="recorder.js"> </script>
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

        var args = {audio: true};
        window.URL = window.URL || window.webkitURL;
        navigator.getUserMedia  = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;
        var recorder = null;
        var recordingTimeout = null;

        window.onload = function() {
            var status = document.querySelector('#status');
            setTimeout(makeActive, 100, 'intro');
            function sendFile(file) {
                var uri = "/index.php";
                var xhr = new XMLHttpRequest();
                var fd = new FormData();

                xhr.open("POST", uri, true);
                xhr.onreadystatechange = function() {
                    if (xhr.readyState == 4 && xhr.status == 200) {
                        imageLength = parseInt(xhr.responseText.substr(0, 8), 10);
                        image = xhr.responseText.substr(8, imageLength);
                        textLength = parseInt(xhr.responseText.substr(8+imageLength, 8), 10);
                        text = xhr.responseText.substr(16+imageLength, textLength);
                        status.innerText = text;
                        document.querySelector('#middle').style.backgroundImage = 'url(\'' + image + '\')';
                        document.querySelector('#form').style.height = document.querySelector('#record').scrollHeight;
                        window.inProgressXHR = null;
                    }
                };
                fd.append('myFile', file);
                // Initiate a multipart/form-data upload
                xhr.send(fd);
                if (window.inProgressXHR != null)
                    window.inProgressXHR.abort();
                window.inProgressXHR = xhr;
            }

            function animate(s) {
                notefg = document.querySelector('#notefg');
                notefg.className = notefg.className.replace(/(?:^|\s)notefg\S*(?!\S)/g, '');
                notefg.className += ' ' + s;
                notefg.parentNode.replaceChild(notefg.cloneNode(true), notefg);
            }

            function record() {
                if (recorder != null) {
                    recorder.clear();
                    recorder.record();
                    animate('notefganim');
                    status.innerText = 'Recording...';
                    if (recordingTimeout != null)
                        window.clearTimeout(recordingTimeout);
                    recordingTimeout = window.setTimeout(stopRecording, 5000);
                }
            }

			var onSuccess = function(s) {
				var context = new webkitAudioContext();
				var mediaStreamSource = context.createMediaStreamSource(s);
				recorder = new Recorder(mediaStreamSource);
                record();
			}

			var onFail = function(e) {
                status.innerText = 'Cannot record audio';
                animate('notefgfailanim');
			};

            function startRecording() {
                if (recorder == null) {
                    if (navigator.getUserMedia) {
                        navigator.getUserMedia(args, onSuccess, onFail);
                    } else {
                        status.innerText = 'Browser not supported';
                        animate('notefgfailanim');
                    }
                } else {
                    record();
                }
                makeActive('record');
            }
            window.startRecording = startRecording;

            function stopRecording() {
                if (recordingTimeout != null)
                    window.clearTimeout(recordingTimeout);
                recordingTimeout = null;
                if (recorder != null) {
                    recorder.stop();
                    recorder.exportWAV(function(s) {
                        sendFile(s);
                    });
                    recorder.clear();
                }
                status.innerText = 'Uploading...';
            }

            function cancelRecord() {
                makeActive('intro')
                if (recordingTimeout != null)
                    window.clearTimeout(recordingTimeout);
                recordingTimeout = null;
                if (recorder != null) {
                    recorder.stop();
                    recorder.clear();
                }
                if (window.inProgressXHR != null) {
                    window.inProgressXHR.abort();
                    window.inProgressXHR = null;
                }
            }
            window.cancelRecord = cancelRecord;
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
                                <input class="submit-button" type="button" value="I'm in the building" onclick="window.startRecording();" style="margin-bottom:5px;"/><br/>
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
                            <div class="noteshadow">&#x266C;</div>
                            <div class="note">&#x266C;</div>
                            <div id="notefg">&#x266C;</div>
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
            <text x="0" y="140" font-family="'Helvetica Neue',Helvetica,sans-serif" font-size="140">&#x266C;</text>
          </clipPath>
        </svg>
    </body>
</html>

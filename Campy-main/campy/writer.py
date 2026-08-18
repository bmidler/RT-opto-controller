"""
"""
from imageio_ffmpeg import write_frames
import os, sys, time, logging
from campy.utils.utils import QueueKeyboardInterrupt
from datetime import datetime

datestr = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

def OpenWriter(cam_params, queue):
	try:

		writing = False
		session_name = os.path.join(cam_params["videoFolder"], str(cam_params["videoFilename"][0:-4]))
		folder_name = os.path.join(session_name, cam_params["cameraName"])
		if cam_params["videoFilename"][-15:-4] == "calibration":
			calibration_folder_name = os.path.join(session_name, folder_name, str('calibration_images'))
		file_name = cam_params["videoFilename"][0:-4]

		# New filename method that puts "-calibration" in the right place.
		fname_prefix = '-{qp' + str(cam_params["quality"]) + '-' + str(cam_params["frameRate"]).zfill(3) + 'fps}'
		file_name = '{' + datestr + '}' + fname_prefix + '-s#' + cam_params["cameraSerialNo"][-4:] + '-' + cam_params['cameraName'] + "-" + file_name + str(cam_params["videoFilename"][-4:])
		
		if cam_params["videoFilename"][-15:-4] == "calibration":
			full_file_name = os.path.join(session_name, folder_name, calibration_folder_name, file_name)
		else: full_file_name = os.path.join(session_name, folder_name, file_name) 

		### Un-breaks metadata save-out.

		# Make sure the file type mp4 is in the file name.
		if ".mp4" not in cam_params["videoFilename"]:
			raise Exception("You're missing the .mp4 file type from the file name in the config file.")
		
		# Add new term to params dict for the modified folder name.
		if cam_params["videoFilename"][-15:-4] == "calibration":
			cam_params["ModVideoFolder"] = os.path.join(session_name, folder_name, calibration_folder_name)
		else:
			cam_params["ModVideoFolder"] = os.path.join(session_name, folder_name)

		### Resume normal writer functions.

		if not os.path.isdir(session_name):
			os.makedirs(session_name, exist_ok=True)
			print("Made directory {}.".format(session_name))

		if not os.path.isdir(folder_name):
			os.makedirs(folder_name, exist_ok=True)
			print("Made directory {}.".format(folder_name))

		if cam_params["videoFilename"][-15:-4] == "calibration":
			if not os.path.isdir(calibration_folder_name):
				os.makedirs(calibration_folder_name, exist_ok=True)
				print("Made directory {}.".format(calibration_folder_name))

		# Flip blue and red for flir camera input
		if cam_params["pixelFormatInput"] == "bayer_bggr8" and cam_params["cameraMake"] == "flir":
			cam_params["pixelFormatInput"] == "bayer_rggb8"

		# Load encoding parameters from cam_params
		pix_fmt_out = cam_params["pixelFormatOutput"]
		codec = str(cam_params["codec"])
		quality = str(cam_params["quality"])
		preset = str(cam_params["preset"])
		frameRate = str(cam_params["frameRate"])
		gpuID = str(cam_params["gpuID"])

		# Load defaults
		gpu_params = []

		# CPU compression
		if cam_params["gpuID"] == -1:
			print("Opened: {} using CPU to compress the stream.".format(full_file_name))
			if preset == "None":
				preset = "fast"
			gpu_params = [
				"-preset", preset,
				"-tune", "fastdecode",
				"-crf", quality,
				"-bufsize", "20M",
				"-maxrate", "10M",
				"-bf:v", "4",
				]
			if pix_fmt_out == "rgb0" or pix_fmt_out == "bgr0":
				pix_fmt_out = "yuv420p"
			if cam_params["codec"] == "h264":
				codec = "libx264"
				gpu_params.append("-x264-params")
				gpu_params.append("nal-hrd=cbr")
			elif cam_params["codec"] == "h265":
				codec = "libx265"

		# GPU compression
		else:
			# Nvidia GPU (NVENC) encoder optimized parameters
			print("Opened: {} using GPU {} to compress the stream.".format(full_file_name, cam_params["gpuID"]))
			if cam_params["gpuMake"] == "nvidia":
				if preset == "None":
					preset = "fast"
				gpu_params = [
					"-preset", preset, # set to "fast", "llhp", or "llhq" for h264 or hevc
					"-qp", quality,
					"-bf:v", "0",
					"-gpu", gpuID,
					"-vsync", "0",
					#"-init_hw_device", "cuda=cu:0",
					#"-filter_hw_device", "cu",
					#"-vf","atadenoise=s=5",
					#"-vf","atadenoise=s=5",
					#"-hwaccel", "cuda"
					#"-fps_mode","passthrough",
					#"-vf","atadenoise=0.02:0.02:0.02:0.04:0.02:0.04:5:all:p",
					#"-filter:v", "atadenoise=0a=0.02:0b=0.04:1a=0.02:1b=0.04:2a=0.02:2b=0.04:s=9:p=7:a='p':0s=32767:1s=32767:2s=32767",
					]
				if cam_params["codec"] == "h264":
					codec = "h264_nvenc"
				elif cam_params["codec"] == "h265": #"h265"
					codec = "hevc_nvenc" #codec = "hevc_nvenc"
				#print(cam_params)
				#print(gpu_params)


			# AMD GPU (AMF/VCE) encoder optimized parameters
			elif cam_params["gpuMake"] == "amd":
				# Preset not supported by AMF
				gpu_params = [
					"-usage", "lowlatency",
					"-rc", "cqp", # constant quantization parameter
					"-qp_i", quality,
					"-qp_p", quality,
					"-qp_b", quality,
					"-bf:v", "0",
					"-hwaccel_device", gpuID,]
				if pix_fmt_out == "rgb0" or pix_fmt_out == "bgr0":
					pix_fmt_out = "yuv420p"
				if cam_params["codec"] == "h264":
					codec = "h264_amf"
				elif cam_params["codec"] == "h265":
					codec = "hevc_amf"

			# Intel iGPU encoder (Quick Sync) optimized parameters				
			elif cam_params["gpuMake"] == "intel":
				if preset == "None":
					preset = "faster"
				gpu_params = [
						"-bf:v", "0",
						"-preset", preset,
						"-q", str(int(quality)+1),]
				if pix_fmt_out == "rgb0" or pix_fmt_out == "bgr0":
					pix_fmt_out = "nv12"
				if cam_params["codec"] == "h264":
					codec = "h264_qsv"
				elif cam_params["codec"] == "h265":
					codec = "hevc_qsv"

	except Exception as e:
		logging.error("Caught exception at writer.py OpenWriter: {}".format(e))
		raise

	# Initialize writer object (imageio-ffmpeg)
	while(True):
		try:
			writer = write_frames(
				full_file_name,
				[cam_params["frameWidth"], cam_params["frameHeight"]], # size [W,H]
				fps=cam_params["frameRate"],
				quality=None,
				codec=codec,
				pix_fmt_in=cam_params["pixelFormatInput"], # "bayer_bggr8", "gray", "rgb24", "bgr0", "yuv420p"
				pix_fmt_out=pix_fmt_out,
				bitrate=None,
				ffmpeg_log_level=cam_params["ffmpegLogLevel"], # "warning", "quiet", "info"
				input_params=["-an"], # "-an" no audio
				output_params=gpu_params,
				)
			writer.send(None) # Initialize the generator
			writing = True
			break
			
		except Exception as e:
			logging.error("Caught exception at writer.py OpenWriter: {}".format(e))
			raise
			break

	# Initialize read queue object to signal interrupt
	readQueue = {}
	readQueue["queue"] = queue
	readQueue["message"] = "STOP"

	return writer, writing, readQueue

def WriteFrames(cam_params, writeQueue, stopReadQueue, stopWriteQueue):
	# Start ffmpeg video writer 
	writer, writing, readQueue = OpenWriter(cam_params, stopReadQueue)

	with QueueKeyboardInterrupt(readQueue):
		# Write until interrupted and/or stop message received
		while(writing):
			if writeQueue:
				writer.send(writeQueue.popleft())
			else:
				# Once queue is depleted and grabber stops, then stop writing
				if stopWriteQueue:
					writing = False
				# Otherwise continue writing
				time.sleep(0.01)

	# Close up...
	print("Closing video writer for {}. Please wait...".format(cam_params["cameraName"]))
	time.sleep(1)
	writer.close()
    


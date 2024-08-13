
def PathMaker():
    import os

    origin_path = os.getcwd()
    path = os.path.join(origin_path, 'log')
    return path
def log():
    # Logging 모듈은 파이썬 기본 라이브러리 중 하나로 콘솔에 출력할 뿐만 아니라 파일 형태로 로그를 생성 할 수 있습니다.
    import logging, os

    # 1. logger instance 설정
    logger = logging.getLogger(__name__)

    # 2. formatter 생성 (로그 출력/저장에 사용할 날짜 + 로그 메시지)
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # 3. handler 생성 (streamHandler : 콘솔 출력용 // fileHandler : 파일 기록용)
    file_path = PathMaker()
    streamHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler(os.path.join(file_path, 'server.log'))    # 로그를 기록할 파일 이름 지정

    # 4. logger instance에 formatter 설정 (각각의 Handler에 formatter 설정 적용)
    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    # 5. logger instance에 handler 추가 (입력받는 log에 handler사용)
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)

    # 6. 기록할 log level 지정하기
    logger.setLevel(level=logging.DEBUG)    # INFO 레벨로 지정하면, INFO 레벨보다 낮은 DEBUG 로그는 무시함.
                                            # Python의 기본 logging 시스템의 레벨은 WARNING으로 설정되어 있음.
                                            # 따라서 특별한 설정을 하지 않으면, WARNING 레벨 이상만 기록됨.

    # 설정된 log setting 반환
    return logger

# 7. log 함수 선 호출 (함수를 한번만 호출해 놓으면, 이후에 logger.debug("메시지") 형식으로 필요 시마다 간단하게 로그를 기록할 수 있음.
logger=log()

# 8. log 기록
# 프로그램 시작 시간 기록
# logger.debug, logger.info, logger.warning, logger.error, logger.critical 사용가능
logger.debug("==========PROGRAM START==========")


# 프로그램 종료 시간 기록
logger.debug("==========PROGRAM FINISH==========")

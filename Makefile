run:
	@python Stock_Price_Prediction_package_folder/main.py

run_uvicorn:
	@uvicorn Stock_Price_Prediction_package_folder.api:app --reload

install:
	@pip install -e .

test:
	@pytest -v tests

reset_processed_data_dir:
	@rm -rf ${PROCESSED_DATA_DIR}
	@mkdir ${PROCESSED_DATA_DIR}

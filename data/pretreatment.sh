for scene in 'chess' 'fire' 'heads' 'office' 'pumpkin' 'redkitchen' 'stairs'
do
echo "--------- Start standardizing  $scene ---------"
python dataset_standard.py --dataset Cambridge --scene $scene
done
echo "--------- Complete standardization ---------"


for scene in 'KingsCollege' 'OldHospital' 'ShopFacade' 'StMarysChurch'
do
echo "--------- Start standardizing  $scene ---------"
python dataset_standard.py --dataset Cambridge --scene $scene
done
echo "--------- Complete standardization ---------"



for scene in 'KingsCollege' 'OldHospital' 'ShopFacade' 'StMarysChurch'
do
echo "--------- start resizing $scene ---------"
python resize_cambridge.py --dataset Cambridge --scene $scene
done
echo "---------- Resize completed ----------"
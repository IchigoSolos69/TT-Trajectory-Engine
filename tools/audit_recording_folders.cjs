'use strict';
const {main}=require('./recording_folder_inventory.cjs');
main().catch(error=>{console.error(error.stack);process.exitCode=1;});
